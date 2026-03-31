"""PyTorchマルチタスク学習モデル (v2: Focal Loss + Dropout低減)"""
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BoatraceMultiTaskModel(nn.Module):
    """マルチタスク学習モデル: 1着/2着/3着を同時予測

    入力: 208次元 (グローバル16 + 艇別32×6)
    隠れ層: [512, 256, 128]
    出力: 6ユニット×3ヘッド

    v2変更: dropout 0.3 → 0.15 (確率分布の平坦化を防止)
    """

    def __init__(self, input_dim=208, hidden_dims=None, num_boats=6,
                 dropout=0.15):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_boats = num_boats
        self.dropout = dropout

        # 共有特徴抽出層
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.shared = nn.Sequential(*layers)

        # タスク別出力ヘッド
        self.head_1st = nn.Linear(hidden_dims[-1], num_boats)
        self.head_2nd = nn.Linear(hidden_dims[-1], num_boats)
        self.head_3rd = nn.Linear(hidden_dims[-1], num_boats)

    def forward(self, x):
        shared_out = self.shared(x)
        out_1st = self.head_1st(shared_out)
        out_2nd = self.head_2nd(shared_out)
        out_3rd = self.head_3rd(shared_out)
        return out_1st, out_2nd, out_3rd


class FocalLoss(nn.Module):
    """Focal Loss + Label Smoothing: 難しいケースに集中 & 過信を抑制

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=0.0 → 通常の CrossEntropyLoss と同等
    gamma=2.0 → 標準的な Focal Loss (推奨)

    label_smoothing: 0.0=ハードラベル, 0.1=推奨
        [1,0,0,0,0,0] → [0.9, 0.02, 0.02, 0.02, 0.02, 0.02]
        1号艇退化解を防ぐ: 100%確信のラベルを与えない

    class_weights: クラス重みテンソル (shape=[num_classes])
        None の場合は均等重み。
    """

    def __init__(self, gamma=2.0, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer(
            'class_weights',
            class_weights if class_weights is not None else None
        )

    def forward(self, logits, targets):
        """
        logits:  (batch_size, num_classes) - softmax前の生出力
        targets: (batch_size,) - 正解クラスインデックス
        """
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Label Smoothing: ハードラベルを軟化
        if self.label_smoothing > 0:
            targets_smooth = F.one_hot(targets, num_classes=num_classes).float()
            targets_smooth = (1.0 - self.label_smoothing) * targets_smooth + \
                             self.label_smoothing / num_classes
        else:
            targets_smooth = F.one_hot(targets, num_classes=num_classes).float()

        # 正解クラスの確率を取得（Focal重み用）
        pt = (probs * targets_smooth).sum(dim=1)  # (batch_size,)

        # Focal 重み: (1 - p_t)^gamma
        focal_weight = (1.0 - pt) ** self.gamma

        # クラス重み
        if self.class_weights is not None:
            alpha = self.class_weights[targets]
        else:
            alpha = 1.0

        # Smoothed Cross Entropy
        loss_per_sample = -(targets_smooth * log_probs).sum(dim=1)
        loss = alpha * focal_weight * loss_per_sample
        return loss.mean()


class BoatraceMultiTaskLoss(nn.Module):
    """マルチタスク加重損失 (v3: Focal Loss + Label Smoothing + 逆頻度クラス重み)

    v3変更:
    - 1着ヘッド: 逆頻度クラス重み (1号艇55%→重み低、5-6号艇→重み高)
    - Label Smoothing: 1着ヘッドに適用 (退化解を防ぐ)
    - 2着/3着ヘッド: クラス重みあり (smoothing=0.7で軽い補正)
    - タスク重み: [1.0, 0.7, 0.5] (変更なし)
    """

    def __init__(self, weights=None,
                 class_weights_1st=None,
                 class_weights_2nd=None,
                 class_weights_3rd=None,
                 gamma=2.0,
                 label_smoothing_1st=0.0):
        super().__init__()
        self.weights = weights or [1.0, 0.7, 0.5]
        self.criterion_1st = FocalLoss(gamma=gamma, class_weights=class_weights_1st,
                                       label_smoothing=label_smoothing_1st)
        self.criterion_2nd = FocalLoss(gamma=gamma, class_weights=class_weights_2nd)
        self.criterion_3rd = FocalLoss(gamma=gamma, class_weights=class_weights_3rd)

    def forward(self, outputs, targets):
        loss_1st = self.criterion_1st(outputs[0], targets[0])
        loss_2nd = self.criterion_2nd(outputs[1], targets[1])
        loss_3rd = self.criterion_3rd(outputs[2], targets[2])
        return (self.weights[0] * loss_1st +
                self.weights[1] * loss_2nd +
                self.weights[2] * loss_3rd)


def save_model(model, path='models/boatrace_model.pth', metadata=None):
    """モデルを保存"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'model_state_dict': model.state_dict(),
        'input_dim': model.input_dim,
        'hidden_dims': model.hidden_dims,
        'num_boats': model.num_boats,
        'dropout': model.dropout,
    }
    if metadata:
        state['metadata'] = metadata
    torch.save(state, path)
    logger.info(f"モデル保存: {path}")


def load_model(path='models/boatrace_model.pth', device=None):
    """モデルを読み込み"""
    if device is None:
        device = torch.device('cpu')
    state = torch.load(path, map_location=device, weights_only=False)
    model = BoatraceMultiTaskModel(
        input_dim=state.get('input_dim', 208),
        hidden_dims=state.get('hidden_dims', [512, 256, 128]),
        num_boats=state.get('num_boats', 6),
        dropout=state.get('dropout', 0.15),
    )
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    logger.info(f"モデル読み込み: {path}")
    return model
