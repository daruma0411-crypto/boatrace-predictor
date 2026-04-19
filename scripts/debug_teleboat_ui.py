"""テレボートSP版の現行UIを解析するデバッグスクリプト

ログイン → トップ → 開催場 → 投票画面まで到達してDOMとスクショを取得。
既存src/teleboat.pyのセレクタがどこで壊れたか特定する。
"""
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv()

SS_DIR = Path(__file__).parent / "teleboat_screenshots"
DEBUG_DIR = Path(__file__).parent / "teleboat_ui_debug"
DEBUG_DIR.mkdir(exist_ok=True)


async def main():
    from playwright.async_api import async_playwright

    member_id = os.environ["TELEBOAT_MEMBER_ID"]
    pin = os.environ["TELEBOAT_PIN"]
    auth = os.environ["TELEBOAT_AUTH"]

    pw = await async_playwright().start()
    iphone = pw.devices['iPhone 12 Pro']
    browser = await pw.webkit.launch(headless=True)
    context = await browser.new_context(**iphone)
    page = await context.new_page()
    page.set_default_timeout(15000)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    async def snap(label):
        p = DEBUG_DIR / f"{ts}_{label}.png"
        await page.screenshot(path=str(p), full_page=True)
        return p

    async def dump_html(label):
        p = DEBUG_DIR / f"{ts}_{label}.html"
        html = await page.content()
        p.write_text(html, encoding="utf-8")
        return p

    print("[1] トップアクセス")
    await page.goto("https://spweb.brtb.jp/", wait_until="networkidle")
    await asyncio.sleep(2)
    await snap("01_top_before_login")

    print("[2] ログインボタン")
    login_div = await page.query_selector("div.btn-login")
    if not login_div:
        print("  ✗ div.btn-login なし")
        await snap("02_no_login_btn")
        await browser.close()
        await pw.stop()
        return
    await login_div.click()

    await page.wait_for_url("**/login.brtb.jp/**", timeout=15000)
    await asyncio.sleep(1)

    print("[3] 認証情報入力")
    await page.fill('input[name="memberNo"]', member_id)
    await page.fill('input[name="pin"]', pin)
    await page.fill('input[name="authNo"]', auth)
    await page.click("button#lbtn")

    await page.wait_for_url("**/spweb.brtb.jp/**", timeout=20000)
    await asyncio.sleep(2)

    # モーダル閉じ
    await page.evaluate(
        'document.querySelectorAll('
        '"[class*=modal],[class*=overlay],[class*=error-block],.l-error-block"'
        ').forEach(e=>e.style.display="none")'
    )
    await asyncio.sleep(1)

    # /top 遷移
    if "/top" not in page.url:
        print("[3.5] /top 遷移試行")
        await page.evaluate("""
            () => {
                const els = document.querySelectorAll(".menu-list-link, a, div");
                for (const el of els) {
                    if (el.textContent.trim() === "トップページ") { el.click(); return; }
                }
            }
        """)
        await asyncio.sleep(3)

    await snap("03_after_login")
    await dump_html("03_after_login")
    print(f"  URL: {page.url}")

    # 開催場パネル探索
    print("[4] 開催場パネル検索")
    panels = await page.query_selector_all("div.jyo-panel")
    print(f"  div.jyo-panel: {len(panels)}件")

    active_venue = None
    active_venue_name = None
    for i, panel in enumerate(panels):
        cls = await panel.get_attribute("class") or ""
        name_el = await panel.query_selector(".jyo-panel-name")
        name = (await name_el.inner_text()).strip() if name_el else "?"
        is_disabled = "is-disabled" in cls
        print(f"  [{i+1}] {name} | disabled={is_disabled} | class={cls[:60]}")
        if not is_disabled and active_venue is None:
            active_venue = panel
            active_venue_name = name

    if active_venue is None:
        print("  ✗ 開催中の場なし（土曜深夜のため）")
        print("  → トップ画面のDOM解析は完了。投票画面は昼間に再実行が必要")
        await browser.close()
        await pw.stop()
        return

    print(f"[5] 場選択: {active_venue_name}")
    await active_venue.evaluate("el => el.click()")
    await asyncio.sleep(3)
    await snap("05_venue_selected")
    await dump_html("05_venue_selected")
    print(f"  URL: {page.url}")

    # レース番号ボタン
    print("[6] レース選択ボタン")
    race_btn = await page.query_selector("button.btn-select-race")
    if race_btn:
        text = (await race_btn.inner_text()).strip()
        print(f"  現在レース: {text}")
    else:
        print("  ✗ button.btn-select-race なし")

    # 勝式セレクタ
    print("[7] 勝式セレクタ")
    selectboxes = await page.query_selector_all("div.selectbox")
    print(f"  div.selectbox: {len(selectboxes)}件")
    for i, sb in enumerate(selectboxes):
        text = (await sb.inner_text()).strip()
        print(f"  [{i+1}] {text[:50]}")

    # ベットラベル
    print("[8] ベットラベル検出")
    for name in ["bet1", "bet2", "bet3"]:
        for n in range(1, 7):
            sel = f'label[for="{name}-{n}"]'
            exists = await page.query_selector(sel)
            if exists and n == 1:
                print(f"  {sel}: 存在")

    # 着順選択試行: 1-2-3
    print("[9] 着順選択: 1-2-3")
    for name, num in [("bet1", 1), ("bet2", 2), ("bet3", 3)]:
        clicked = await page.evaluate(
            f'(s) => {{ const el = document.querySelector(s); if (el) {{ el.click(); return true; }} return false; }}',
            f'label[for="{name}-{num}"]'
        )
        print(f"  label[for={name}-{num}]: clicked={clicked}")
        await asyncio.sleep(0.3)

    # 金額入力
    print("[10] 金額入力フィールド検出")
    amt = await page.query_selector('input[type="tel"].textbox')
    print(f"  input[type=tel].textbox: {'あり' if amt else 'なし'}")
    if not amt:
        amt = await page.query_selector('input.textbox')
        print(f"  input.textbox: {'あり' if amt else 'なし'}")
    if amt:
        await amt.fill("1")

    await snap("10_bets_selected")
    await dump_html("10_bets_selected")

    # === 問題の「投票へ進む」ボタン探索 ===
    print("\n[11] 投票へ進むボタン徹底探索")

    # 方法1: 既存コード (button/div[class*='btn'])
    found_existing = None
    buttons1 = await page.query_selector_all("button, div[class*='btn']")
    print(f"  方法1 (button, div[class*='btn']): {len(buttons1)}件検出")
    for btn in buttons1:
        text = (await btn.inner_text()).strip()
        if "投票へ進む" in text or "投票に進む" in text:
            found_existing = btn
            print(f"    → HIT: '{text[:40]}'")
            tag = await btn.evaluate("el => el.tagName")
            cls = await btn.get_attribute("class") or ""
            print(f"      tag={tag} class={cls[:80]}")
            break
    if not found_existing:
        print("  ✗ 方法1で投票ボタン未検出")

    # 方法2: より広いセレクタ
    print("\n  方法2: 全click-able要素スキャン")
    all_clickable = await page.query_selector_all(
        "button, a, div[role='button'], input[type='button'], input[type='submit'], "
        "[onclick], [class*='button'], [class*='btn'], label"
    )
    print(f"    候補: {len(all_clickable)}件")
    vote_candidates = []
    for el in all_clickable:
        try:
            text = (await el.inner_text()).strip()
        except Exception:
            continue
        if any(k in text for k in ["投票", "進む", "ベット", "確定", "追加"]):
            tag = await el.evaluate("el => el.tagName")
            cls = await el.get_attribute("class") or ""
            visible = await el.is_visible()
            vote_candidates.append({"text": text[:40], "tag": tag, "class": cls[:100], "visible": visible})
    for c in vote_candidates:
        print(f"    - [{c['tag']}] visible={c['visible']} '{c['text']}' class={c['class']}")

    # 方法3: JSで全テキスト解析
    print("\n  方法3: JS evaluate で「投票」含むclick-able要素")
    js_result = await page.evaluate("""
        () => {
            const els = document.querySelectorAll('*');
            const results = [];
            for (const el of els) {
                const text = el.innerText || '';
                if (text.trim() === '' || text.length > 50) continue;
                if (text.includes('投票へ進む') || text.includes('投票に進む') || text.includes('ベットリストに追加')) {
                    const rect = el.getBoundingClientRect();
                    results.push({
                        tag: el.tagName,
                        text: text.trim().substring(0, 50),
                        class: el.className || '',
                        id: el.id || '',
                        visible: rect.width > 0 && rect.height > 0,
                    });
                }
            }
            return results.slice(0, 10);
        }
    """)
    for r in js_result:
        print(f"    - [{r['tag']}] visible={r['visible']} id='{r['id']}' class='{r['class']}' text='{r['text']}'")

    await snap("11_final")

    # === 「投票へ進む」クリック → ベットリスト画面 ===
    if found_existing:
        print("\n[12] 投票へ進むクリック")
        await found_existing.evaluate("el => el.click()")
        await asyncio.sleep(3)
        await snap("12_after_vote_btn")
        await dump_html("12_after_vote_btn")
        print(f"  URL: {page.url}")

        # 「次へ」ボタン探索
        print("[13] 次へボタン探索")
        buttons = await page.query_selector_all("button, div[class*='btn'], a[class*='btn']")
        next_candidates = []
        for btn in buttons:
            try:
                text = (await btn.inner_text()).strip()
            except Exception:
                continue
            if text == "次へ" or "次へ" in text:
                tag = await btn.evaluate("el => el.tagName")
                cls = await btn.get_attribute("class") or ""
                visible = await btn.is_visible()
                next_candidates.append({"text": text[:30], "tag": tag, "class": cls[:80], "visible": visible})
        for c in next_candidates:
            print(f"    - [{c['tag']}] visible={c['visible']} '{c['text']}' class={c['class']}")

        next_btn = None
        for btn in buttons:
            try:
                text = (await btn.inner_text()).strip()
            except Exception:
                continue
            if text == "次へ":
                next_btn = btn
                break

        if next_btn:
            print("  → 次へクリック")
            await next_btn.evaluate("el => el.click()")
            await asyncio.sleep(3)
            await snap("13_after_next")
            await dump_html("13_after_next")
            print(f"  URL: {page.url}")

            # 確認画面: 合計金額入力 + 投票ボタン
            print("[14] 合計金額入力フィールド")
            total_input = await page.query_selector('input[type="tel"].textbox')
            if not total_input:
                total_input = await page.query_selector('input.textbox')
            if total_input:
                print("  → input.textbox 発見")
                await total_input.fill("100")
                await asyncio.sleep(0.5)
                await snap("14_total_filled")
            else:
                print("  ✗ 合計金額input未検出")

            print("[15] 投票ボタン探索")
            all_buttons = await page.query_selector_all("button, div[class*='btn'], a[class*='btn']")
            vote_candidates2 = []
            for btn in all_buttons:
                try:
                    text = (await btn.inner_text()).strip()
                except Exception:
                    continue
                if text in ("投票", "投票する") or "投票" in text:
                    tag = await btn.evaluate("el => el.tagName")
                    cls = await btn.get_attribute("class") or ""
                    visible = await btn.is_visible()
                    disabled = await btn.get_attribute("disabled")
                    aria = await btn.get_attribute("aria-disabled")
                    vote_candidates2.append({"text": text[:30], "tag": tag, "class": cls[:80],
                                             "visible": visible, "disabled": disabled, "aria": aria})
            for c in vote_candidates2:
                print(f"    - [{c['tag']}] visible={c['visible']} disabled={c['disabled']} aria={c['aria']} '{c['text']}' class={c['class']}")

            # **ここまででDRY_RUN終了 — 実投票はしない**
            print("\n[16] === DRY RUN: 実投票はスキップ ===")
            await snap("16_dryrun_stop")
        else:
            print("  ✗ 次へボタン未検出")

    print(f"\n=== 完了 ===")
    print(f"SS/HTML保存先: {DEBUG_DIR}")

    await browser.close()
    await pw.stop()


if __name__ == '__main__':
    asyncio.run(main())
