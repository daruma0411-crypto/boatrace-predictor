"""テレボートSP版 (spweb.brtb.jp) Playwright自動操作モジュール

WebKit (Safari) エミュレーション (iPhone 12 Pro) でテレボートSP版にアクセスし、
舟券購入を自動化する。Chromiumではデバイスチェックに弾かれるためWebKit必須。

ログインフロー:
    spweb.brtb.jp → div.btn-login →
    login.brtb.jp → memberNo/pin/authNo → button#lbtn →
    spweb.brtb.jp/top にリダイレクト

投票フロー:
    トップ(ul.jyo-list) → div.jyo-panel クリック →
    /bet ページ → 場/レース切替(btn-select-m / btn-select-race) →
    着順チェックボックス(input[name=bet1/bet2/bet3]) →
    金額入力(input[type=tel].textbox) →
    「ベットリストに追加して投票へ進む」→ 確認 → 投票確定
"""
import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SS_DIR = Path(__file__).parent.parent / "scripts" / "teleboat_screenshots"
TELEBOAT_SP_URL = "https://spweb.brtb.jp/"

VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島',
    5: '多摩川', 6: '浜名湖', 7: '蒲郡', 8: '常滑',
    9: '津', 10: '三国', 11: 'びわこ', 12: '住之江',
    13: '尼崎', 14: '鳴門', 15: '丸亀', 16: '児島',
    17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}


class TelebotPurchaser:
    """テレボートSP版 Playwright自動操作 (WebKit/Safari)

    Usage:
        purchaser = TelebotPurchaser(member_id, pin, auth_number, dry_run=True)
        await purchaser.start()
        await purchaser.login()
        result = await purchaser.purchase(venue_id=12, race_number=5,
                                          combination="1-2-3", amount=500)
        balance = await purchaser.get_balance()
        await purchaser.close()
    """

    def __init__(self, member_id, pin, auth_number, dry_run=False):
        self.member_id = member_id
        self.pin = pin
        self.auth_number = auth_number
        self.dry_run = dry_run
        self.browser = None
        self.context = None
        self.page = None
        self._logged_in = False

    async def start(self):
        """WebKit (Safari) ブラウザ起動 + iPhone 12 Pro エミュレーション"""
        from playwright.async_api import async_playwright
        self._pw = await async_playwright().start()

        iphone = self._pw.devices['iPhone 12 Pro']
        self.browser = await self._pw.webkit.launch(headless=True)
        self.context = await self.browser.new_context(**iphone)
        self.page = await self.context.new_page()

        self.page.set_default_timeout(15000)
        self.page.set_default_navigation_timeout(30000)

        SS_DIR.mkdir(exist_ok=True)
        logger.info("WebKit (Safari) ブラウザ起動完了")

    async def _screenshot(self, name):
        """スクリーンショット保存"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SS_DIR / f"{ts}_{name}.png"
        await self.page.screenshot(path=str(path), full_page=True)
        logger.debug(f"SS: {path.name}")
        return str(path)

    async def _wait_stable(self, timeout=10000):
        """ページ安定待ち"""
        try:
            await self.page.wait_for_load_state("networkidle", timeout=timeout)
        except Exception:
            pass
        await asyncio.sleep(1)

    async def _close_modal(self):
        """お知らせモーダルを閉じる"""
        await self.page.evaluate(
            'document.querySelectorAll("[class*=modal],[class*=overlay]")'
            '.forEach(e=>e.style.display="none")'
        )
        await asyncio.sleep(0.5)

    async def login(self):
        """ログイン

        Returns:
            bool: ログイン成功
        """
        logger.info("テレボートログイン開始")

        # Step 1: トップページ → ログインボタン
        await self.page.goto(TELEBOAT_SP_URL, wait_until="networkidle")
        login_div = await self.page.query_selector("div.btn-login")
        if not login_div:
            logger.error("div.btn-login が見つかりません")
            return False
        await login_div.click()

        # Step 2: login.brtb.jp 遷移待ち
        try:
            await self.page.wait_for_url("**/login.brtb.jp/**", timeout=15000)
        except Exception:
            pass
        await self._wait_stable()

        # Step 3: 認証情報入力
        await self.page.fill('input[name="memberNo"]', self.member_id)
        await self.page.fill('input[name="pin"]', self.pin)
        await self.page.fill('input[name="authNo"]', self.auth_number)

        # Step 4: ログインボタン
        lbtn = await self.page.query_selector("button#lbtn")
        if not lbtn:
            logger.error("button#lbtn が見つかりません")
            return False
        await lbtn.click()

        # Step 5: リダイレクト待ち
        try:
            await self.page.wait_for_url("**/spweb.brtb.jp/**", timeout=20000)
        except Exception:
            pass
        await self._wait_stable()

        if "login.brtb.jp" in self.page.url:
            logger.error(f"ログイン失敗: URL={self.page.url}")
            await self._screenshot("login_failed")
            return False

        self._logged_in = True
        await self._close_modal()
        await self._screenshot("login_success")
        logger.info("ログイン成功")
        return True

    async def purchase(self, venue_id, race_number, combination, amount):
        """舟券購入（3連単）

        投票画面の構造:
        - トップ: ul.jyo-list → div.jyo-panel (場名: div.jyo-panel-name)
        - 投票画面: /bet
        - 場切替: button.btn-select-m
        - レース切替: button.btn-select-race
        - 着順: input[name=bet1] (1着), input[name=bet2] (2着), input[name=bet3] (3着)
          各6個（1号艇〜6号艇の順）
        - 金額: input[type=tel].textbox (100円単位の数値)
        - 確定: 「ベットリストに追加して投票へ進む」→「投票」

        Args:
            venue_id: 会場ID (1-24)
            race_number: レース番号 (1-12)
            combination: "1-2-3" 形式
            amount: 金額（円、100円単位）

        Returns:
            dict: {success: bool, message: str, screenshot: str}
        """
        if not self._logged_in:
            return {"success": False, "message": "未ログイン", "screenshot": ""}

        venue_name = VENUE_NAMES.get(venue_id, f"場{venue_id}")
        logger.info(f"購入開始: {venue_name} {race_number}R 3連単 {combination} ¥{amount:,}")

        parts = combination.split("-")
        if len(parts) != 3:
            return {"success": False, "message": f"組合せ形式不正: {combination}", "screenshot": ""}

        first, second, third = [int(p) for p in parts]

        try:
            # --- Step 1: トップ画面 → 開催場クリック ---
            await self.page.goto(TELEBOAT_SP_URL + "top", wait_until="networkidle")
            await self._close_modal()
            await self._screenshot("p01_top")

            venue_clicked = False
            panels = await self.page.query_selector_all("div.jyo-panel")
            for panel in panels:
                cls = await panel.get_attribute("class") or ""
                if "is-disabled" in cls:
                    continue
                name_el = await panel.query_selector(".jyo-panel-name")
                if name_el:
                    name = (await name_el.inner_text()).strip()
                    if name == venue_name:
                        await panel.click()
                        await self._wait_stable()
                        venue_clicked = True
                        logger.info(f"  場選択: {venue_name}")
                        break

            if not venue_clicked:
                ss = await self._screenshot("error_venue")
                return {"success": False, "message": f"場が見つからない/非開催: {venue_name}",
                        "screenshot": ss}

            await self._screenshot("p02_bet_page")

            # --- Step 2: レース切替 ---
            # 現在のレース番号を確認
            race_btn = await self.page.query_selector("button.btn-select-race")
            if race_btn:
                current_race_text = (await race_btn.inner_text()).strip()
                if not current_race_text.startswith(f"{race_number}R"):
                    # レース番号が違うのでドロップダウンから選択
                    await race_btn.click()
                    await asyncio.sleep(1)

                    # ドロップダウンリストからレースを選択
                    race_options = await self.page.query_selector_all(
                        "li[class*='selectbox'], div[class*='selectbox-item']"
                    )
                    for opt in race_options:
                        opt_text = (await opt.inner_text()).strip()
                        if opt_text.startswith(f"{race_number}R"):
                            await opt.click()
                            await self._wait_stable()
                            logger.info(f"  レース切替: {race_number}R")
                            break

            await self._screenshot("p03_race_selected")

            # --- Step 3: 勝式が「3連単」であることを確認 ---
            # デフォルトが3連単でない場合は切替
            bet_type_el = await self.page.query_selector(
                "div.selectbox:has(button:has-text('3連単')), "
                "button:has-text('3連単')"
            )
            if not bet_type_el:
                # 勝式ドロップダウンを探してクリック
                selectboxes = await self.page.query_selector_all("div.selectbox")
                for sb in selectboxes:
                    text = (await sb.inner_text()).strip()
                    if any(k in text for k in ['単', '複', '連', 'ワイド']):
                        # これが勝式ドロップダウン
                        btn = await sb.query_selector("button")
                        if btn:
                            await btn.click()
                            await asyncio.sleep(0.5)
                            # 3連単を選択
                            options = await self.page.query_selector_all("li")
                            for opt in options:
                                if "3連単" in (await opt.inner_text()).strip():
                                    await opt.click()
                                    await self._wait_stable()
                                    logger.info("  勝式: 3連単")
                                    break
                        break

            # --- Step 4: 着順チェックボックス ---
            # input[name=bet1] が6個(1号艇〜6号艇), bet2, bet3 も同様
            # N号艇はN番目(0-indexed: N-1)のチェックボックス

            bet1_inputs = await self.page.query_selector_all('input[name="bet1"]')
            bet2_inputs = await self.page.query_selector_all('input[name="bet2"]')
            bet3_inputs = await self.page.query_selector_all('input[name="bet3"]')

            if len(bet1_inputs) < 6 or len(bet2_inputs) < 6 or len(bet3_inputs) < 6:
                ss = await self._screenshot("error_no_checkboxes")
                return {"success": False,
                        "message": f"チェックボックス不足: bet1={len(bet1_inputs)} bet2={len(bet2_inputs)} bet3={len(bet3_inputs)}",
                        "screenshot": ss}

            # チェックボックスをクリック（labelをクリックする方が確実）
            # 1着: first号艇 (index = first - 1)
            await bet1_inputs[first - 1].click()
            await asyncio.sleep(0.3)
            # 2着: second号艇
            await bet2_inputs[second - 1].click()
            await asyncio.sleep(0.3)
            # 3着: third号艇
            await bet3_inputs[third - 1].click()
            await asyncio.sleep(0.3)

            logger.info(f"  着順選択: {first}-{second}-{third}")
            await self._screenshot("p04_numbers")

            # --- Step 5: 金額入力 ---
            amount_unit = amount // 100
            amount_input = await self.page.query_selector('input[type="tel"].textbox')
            if not amount_input:
                amount_input = await self.page.query_selector('input.textbox')
            if amount_input:
                await amount_input.fill(str(amount_unit))
                logger.info(f"  金額: {amount_unit} (x100 = ¥{amount:,})")
            else:
                logger.warning("  金額入力フィールドが見つかりません")

            await self._screenshot("p05_amount")

            # --- Step 6: 「ベットリストに追加して投票へ進む」 ---
            add_btn = None
            buttons = await self.page.query_selector_all("button, div[class*='btn']")
            for btn in buttons:
                text = (await btn.inner_text()).strip()
                if "投票へ進む" in text or "投票に進む" in text:
                    add_btn = btn
                    break

            if add_btn:
                disabled = await add_btn.get_attribute("disabled")
                if disabled is not None:
                    ss = await self._screenshot("error_btn_disabled")
                    return {"success": False, "message": "投票ボタンが無効（入力不足?）",
                            "screenshot": ss}
                await add_btn.click()
                await self._wait_stable()
                logger.info("  ベットリスト追加 → 投票画面")
            else:
                ss = await self._screenshot("error_no_add_btn")
                return {"success": False, "message": "投票へ進むボタンが見つかりません",
                        "screenshot": ss}

            await self._screenshot("p06_betlist")

            # --- Step 7: 合計金額確認 → 「投票」ボタン ---
            # 投票確認画面で合計金額入力が必要な場合がある
            confirm_input = await self.page.query_selector(
                'input[type="tel"].textbox, input[class*="total"]'
            )
            if confirm_input:
                # 合計金額を入力（Webの仕様に応じて調整）
                current_val = await confirm_input.evaluate("el => el.value")
                if not current_val:
                    await confirm_input.fill(str(amount_unit))

            confirm_ss = await self._screenshot("p07_confirm")

            # --- Step 8: DRY_RUN ---
            if self.dry_run:
                logger.info(f"  [DRY RUN] 購入確定スキップ")
                return {
                    "success": True,
                    "message": f"[DRY RUN] {venue_name} {race_number}R {combination} ¥{amount:,}",
                    "screenshot": confirm_ss,
                }

            # --- Step 9: 「投票」ボタン ---
            vote_btn = None
            buttons = await self.page.query_selector_all("button, div[class*='btn']")
            for btn in buttons:
                text = (await btn.inner_text()).strip()
                if text == "投票" or text == "投票する":
                    vote_btn = btn
                    break

            if vote_btn:
                await vote_btn.click()
                await self._wait_stable()
                logger.info("  投票確定")
            else:
                return {"success": False, "message": "投票ボタンが見つかりません",
                        "screenshot": confirm_ss}

            complete_ss = await self._screenshot("p08_complete")

            page_text = await self.page.inner_text("body")
            if any(k in page_text for k in ["完了", "受付", "投票しました"]):
                msg = f"購入完了: {venue_name} {race_number}R 3連単 {combination} ¥{amount:,}"
                logger.info(f"  {msg}")
                return {"success": True, "message": msg, "screenshot": complete_ss}
            else:
                msg = f"購入結果不明: {venue_name} {race_number}R（SS確認）"
                logger.warning(f"  {msg}")
                return {"success": True, "message": msg, "screenshot": complete_ss}

        except Exception as e:
            error_ss = await self._screenshot("error")
            logger.error(f"  購入エラー: {e}")
            return {"success": False, "message": str(e), "screenshot": error_ss}

    async def get_balance(self):
        """残高確認

        Returns:
            int or None: 残高（円）
        """
        try:
            page_text = await self.page.inner_text("body")
            match = re.search(r'購入残高\s*([\d,]+)\s*円', page_text)
            if match:
                balance = int(match.group(1).replace(',', ''))
                logger.info(f"残高: ¥{balance:,}")
                return balance

            match = re.search(r'残高[^\d]*?([\d,]+)\s*円', page_text)
            if match:
                balance = int(match.group(1).replace(',', ''))
                logger.info(f"残高: ¥{balance:,}")
                return balance

            logger.warning("残高取得失敗")
            return None
        except Exception as e:
            logger.error(f"残高取得エラー: {e}")
            return None

    async def close(self):
        """ブラウザ終了"""
        if self.browser:
            await self.browser.close()
        if hasattr(self, '_pw') and self._pw:
            await self._pw.stop()
        logger.info("ブラウザ終了")
