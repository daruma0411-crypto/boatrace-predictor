"""PyTorchマルチタスク学習モデル"""
import os
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BoatraceMultiTaskModel(nn.Module):
    """マルチタスク学習モデル: 1着/2着/3着を同時予測

    入力: 194次元 (グローバル14 + 艇別30×6)
    隠れ層: [512, 256, 128]
    出力: 6ユニット×3ヘッド
    """

    def __init__(self, input_dim=194, hidden_dims=None, num_boats=6):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_boats = num_boats

        # 共有特徴抽出層
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
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


class BoatraceMultiTaskLoss(nn.Module):
    """マルチタスク加重損失: [1着×1.0, 2着×0.7, 3着×0.5]"""

    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or [1.0, 0.7, 0.5]
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        loss_1st = self.criterion(outputs[0], targets[0])
        loss_2nd = self.criterion(outputs[1], targets[1])
        loss_3rd = self.criterion(outputs[2], targets[2])
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
        input_dim=state.get('input_dim', 194),
        hidden_dims=state.get('hidden_dims', [512, 256, 128]),
        num_boats=state.get('num_boats', 6),
    )
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    logger.info(f"モデル読み込み: {path}")
    return model
