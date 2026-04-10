"""テレボートSP版 (spweb.brtb.jp) Playwright自動操作モジュール

WebKit (Safari) エミュレーション (iPhone 12 Pro) でテレボートSP版にアクセスし、
舟券購入を自動化する。Chromiumではデバイスチェックに弾かれるためWebKit必須。

ログインフロー:
    spweb.brtb.jp → div.btn-login →
    login.brtb.jp → memberNo/pin/authNo → button#lbtn →
    spweb.brtb.jp/top にリダイレクト

投票フロー:
    トップ → 開催場パネル → 投票メニュー →
    着順label(bet1-N/bet2-N/bet3-N) → 金額(100円単位) →
    「投票へ進む」→ ベットリスト「次へ」→ 合計金額入力 → 「投票」確定

2件目以降:
    投票結果「場を変更して投票」→ トップ画面に復帰 → 次の場を選択
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
                                          combination="1-2-3", amount=100)
        # 2件目以降: navigate_to_top()不要（purchase内で自動復帰）
        result2 = await purchaser.purchase(venue_id=6, race_number=3, ...)
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

    async def _navigate_to_top(self):
        """トップ画面（開催場一覧）に復帰

        投票結果画面の「場を変更して投票」ボタン、または
        投票メニューの「トップ」リンクを使う。
        gotoによるセッション切れを回避する。
        """
        # 方法1: 「場を変更して投票」ボタン（投票結果画面）
        buttons = await self.page.query_selector_all("button, a, div[class*='btn']")
        for btn in buttons:
            text = (await btn.inner_text()).strip()
            if "場を変更" in text or "トップへ戻る" in text:
                await btn.click()
                await self._wait_stable()
                await self._close_modal()
                logger.info("  トップ画面に復帰")
                return True

        # 方法2: ヘッダーの「トップ」リンク
        top_link = await self.page.query_selector("a[href*='/top'], a:has-text('トップ')")
        if top_link:
            await top_link.click()
            await self._wait_stable()
            await self._close_modal()
            logger.info("  トップ画面に復帰（ヘッダー）")
            return True

        # 方法3: メニューの「トップページ」をJSクリック
        try:
            clicked = await self.page.evaluate("""
                () => {
                    const els = document.querySelectorAll(".menu-list-link, a, div");
                    for (const el of els) {
                        if (el.textContent.trim() === "トップページ") {
                            el.click(); return true;
                        }
                    }
                    return false;
                }
            """)
            if clicked:
                await self._wait_stable()
                await self._close_modal()
                panels = await self.page.query_selector_all("div.jyo-panel")
                if panels:
                    logger.info("  トップ画面に復帰（JSクリック）")
                    return True
        except Exception:
            pass

        # 方法4: 直接遷移（最終手段）
        try:
            await self.page.goto(TELEBOAT_SP_URL + "top", wait_until="networkidle")
            await self._wait_stable()
            await self._close_modal()
            panels = await self.page.query_selector_all("div.jyo-panel")
            if panels:
                logger.info("  トップ画面に復帰（goto）")
                return True
        except Exception:
            pass

        logger.warning("  トップ画面への復帰に失敗")
        return False

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

        # ログイン後 /top にいない場合は遷移
        if "/top" not in self.page.url:
            # メニュー画面の「トップページ」をJSクリック
            await self.page.evaluate("""
                () => {
                    const els = document.querySelectorAll(".menu-list-link, a, div");
                    for (const el of els) {
                        if (el.textContent.trim() === "トップページ") {
                            el.click(); return;
                        }
                    }
                }
            """)
            await self._wait_stable()
            await self._close_modal()

        await self._screenshot("login_success")
        logger.info("ログイン成功")
        return True

    async def purchase(self, venue_id, race_number, combination, amount):
        """舟券購入（3連単）

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
        amount_unit = amount // 100

        try:
            # --- Step 1: トップ画面 → 開催場クリック ---
            # ログイン直後は /top にいる。2件目以降は _navigate_to_top() で復帰済み。
            # /top にいない場合のみボタン経由で復帰を試みる。
            if "/top" not in self.page.url:
                navigated = await self._navigate_to_top()
                if not navigated:
                    return {"success": False, "message": "トップ画面に戻れません",
                            "screenshot": await self._screenshot("error_nav")}

            await self._close_modal()

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

            # --- Step 2: レース切替 ---
            race_btn = await self.page.query_selector("button.btn-select-race")
            if race_btn:
                current_race_text = (await race_btn.inner_text()).strip()
                if not current_race_text.startswith(f"{race_number}R"):
                    await race_btn.click()
                    await asyncio.sleep(1)

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

            # --- Step 3: 勝式が「3連単」であることを確認 ---
            bet_type_el = await self.page.query_selector(
                "div.selectbox:has(button:has-text('3連単')), "
                "button:has-text('3連単')"
            )
            if not bet_type_el:
                selectboxes = await self.page.query_selector_all("div.selectbox")
                for sb in selectboxes:
                    text = (await sb.inner_text()).strip()
                    if any(k in text for k in ['単', '複', '連', 'ワイド']):
                        btn = await sb.query_selector("button")
                        if btn:
                            await btn.click()
                            await asyncio.sleep(0.5)
                            options = await self.page.query_selector_all("li")
                            for opt in options:
                                if "3連単" in (await opt.inner_text()).strip():
                                    await opt.click()
                                    await self._wait_stable()
                                    logger.info("  勝式: 3連単")
                                    break
                        break

            # --- Step 4: 着順選択 (label経由) ---
            await self.page.click(f'label[for="bet1-{first}"]')
            await asyncio.sleep(0.3)
            await self.page.click(f'label[for="bet2-{second}"]')
            await asyncio.sleep(0.3)
            await self.page.click(f'label[for="bet3-{third}"]')
            await asyncio.sleep(0.3)
            logger.info(f"  着順選択: {first}-{second}-{third}")

            # --- Step 5: 金額入力 (100円単位) ---
            amount_input = await self.page.query_selector('input[type="tel"].textbox')
            if not amount_input:
                amount_input = await self.page.query_selector('input.textbox')
            if amount_input:
                await amount_input.fill(str(amount_unit))
                logger.info(f"  金額: {amount_unit} (x100 = ¥{amount:,})")
            else:
                logger.warning("  金額入力フィールドが見つかりません")

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

            # --- Step 7: ベットリスト画面 →「次へ」→ 確認画面 ---
            next_btn = None
            buttons = await self.page.query_selector_all("button, div[class*='btn'], a[class*='btn']")
            for btn in buttons:
                text = (await btn.inner_text()).strip()
                if text == "次へ":
                    next_btn = btn
                    break

            if next_btn:
                await next_btn.click()
                await self._wait_stable()
                logger.info("  次へ → 確認画面")
            else:
                ss = await self._screenshot("error_no_next_btn")
                return {"success": False, "message": "「次へ」ボタンが見つかりません",
                        "screenshot": ss}

            # --- DRY_RUN チェック (最終確認画面で停止) ---
            if self.dry_run:
                final_ss = await self._screenshot("dryrun_stop")
                logger.info("  [DRY RUN] 投票確定スキップ")
                await self._navigate_to_top()
                return {
                    "success": True,
                    "message": f"[DRY RUN] {venue_name} {race_number}R {combination} ¥{amount:,}",
                    "screenshot": final_ss,
                }

            # --- Step 8: 確認画面 — 合計金額入力 + 「投票」 ---
            total_input = await self.page.query_selector('input[type="tel"].textbox')
            if not total_input:
                total_input = await self.page.query_selector('input.textbox')
            if total_input:
                await total_input.fill(str(amount))
                logger.info(f"  合計金額入力: {amount} 円")
                await asyncio.sleep(0.3)

            vote_btn = None
            buttons = await self.page.query_selector_all("button, div[class*='btn'], a[class*='btn']")
            for btn in buttons:
                text = (await btn.inner_text()).strip()
                if text in ("投票", "投票する"):
                    vote_btn = btn
                    break

            if vote_btn:
                await vote_btn.click()
                await self._wait_stable()
                logger.info("  投票確定")
            else:
                ss = await self._screenshot("error_no_vote_btn")
                return {"success": False, "message": "「投票」ボタンが見つかりません",
                        "screenshot": ss}

            # --- Step 9: 投票結果確認 ---
            complete_ss = await self._screenshot("complete")

            page_text = await self.page.inner_text("body")
            if any(k in page_text for k in ["完了", "受付", "投票しました"]):
                msg = f"購入完了: {venue_name} {race_number}R 3連単 {combination} ¥{amount:,}"
                logger.info(f"  {msg}")
                # 次の購入に備えてトップ画面に復帰
                await self._navigate_to_top()
                return {"success": True, "message": msg, "screenshot": complete_ss}
            else:
                msg = f"購入結果不明: {venue_name} {race_number}R（SS確認）"
                logger.warning(f"  {msg}")
                await self._navigate_to_top()
                return {"success": True, "message": msg, "screenshot": complete_ss}

        except Exception as e:
            error_ss = await self._screenshot("error")
            logger.error(f"  購入エラー: {e}")
            # エラー時もトップ復帰を試みる
            try:
                await self._navigate_to_top()
            except Exception:
                pass
            return {"success": False, "message": str(e), "screenshot": error_ss}

    async def get_balance(self):
        """残高確認

        Returns:
            int or None: 残高（円）
        """
        try:
            page_text = await self.page.inner_text("body")

            # パターン1: 「購入残高 1,000 円」
            match = re.search(r'購入残高\s*([\d,]+)\s*円', page_text)
            if match:
                balance = int(match.group(1).replace(',', ''))
                logger.info(f"残高: ¥{balance:,}")
                return balance

            # パターン2: 「購入残高」の後に数字
            match = re.search(r'購入残高[^\d]*([\d,]+)', page_text)
            if match:
                balance = int(match.group(1).replace(',', ''))
                logger.info(f"残高: ¥{balance:,}")
                return balance

            # パターン3: 「残高」+ 数字 + 「円」
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
