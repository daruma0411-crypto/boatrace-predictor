"""テレボート画面構造探索ツール

Playwrightでテレボートにアクセスし、各画面をスクリーンショット+HTML保存。
自動購入の要素セレクタを特定するための補助ツール。

使い方:
    python scripts/teleboat_discover.py --step login
    python scripts/teleboat_discover.py --step venue
    python scripts/teleboat_discover.py --step race
    python scripts/teleboat_discover.py --step bet
    python scripts/teleboat_discover.py --step all    (全ステップ自動実行)

環境変数:
    TELEBOAT_MEMBER_ID  加入者番号
    TELEBOAT_PIN        暗証番号
    TELEBOAT_AUTH       認証番号
"""
import asyncio
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("playwright が未インストールです。以下を実行してください:")
    print("  pip install playwright")
    print("  playwright install chromium")
    sys.exit(1)

# スクリーンショット保存先
SS_DIR = Path(__file__).parent / "teleboat_screenshots"
SS_DIR.mkdir(exist_ok=True)

# テレボートSP版URL
TELEBOAT_SP_URL = "https://spweb.brtb.jp/"


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


async def save_snapshot(page, step_name):
    """スクリーンショットとHTMLを保存"""
    ts = timestamp()
    ss_path = SS_DIR / f"{ts}_{step_name}.png"
    html_path = SS_DIR / f"{ts}_{step_name}.html"

    await page.screenshot(path=str(ss_path), full_page=True)
    html = await page.content()
    html_path.write_text(html, encoding='utf-8')

    print(f"  [保存] {ss_path.name}")
    print(f"  [保存] {html_path.name}")
    return str(ss_path)


async def create_browser(pw):
    """モバイルエミュレーション (iPhone 12 Pro) でブラウザ起動"""
    iphone = pw.devices['iPhone 12 Pro']
    browser = await pw.chromium.launch(headless=False)  # 探索時はheaded
    context = await browser.new_context(**iphone)
    page = await context.new_page()
    return browser, context, page


async def step_login(page):
    """ログイン画面の探索"""
    print("\n=== Step: login ===")
    member_id = os.environ.get("TELEBOAT_MEMBER_ID", "")
    pin = os.environ.get("TELEBOAT_PIN", "")
    auth = os.environ.get("TELEBOAT_AUTH", "")

    if not all([member_id, pin, auth]):
        print("  [警告] TELEBOAT_MEMBER_ID, TELEBOAT_PIN, TELEBOAT_AUTH を .env に設定してください")
        print("  スクリーンショットのみ取得します")

    # テレボートSP版にアクセス
    print(f"  アクセス: {TELEBOAT_SP_URL}")
    await page.goto(TELEBOAT_SP_URL, wait_until="networkidle", timeout=30000)
    await save_snapshot(page, "01_login_page")

    # フォーム要素の探索
    print("\n  --- フォーム要素探索 ---")
    inputs = await page.query_selector_all("input")
    for i, inp in enumerate(inputs):
        inp_type = await inp.get_attribute("type") or "text"
        inp_name = await inp.get_attribute("name") or ""
        inp_id = await inp.get_attribute("id") or ""
        inp_placeholder = await inp.get_attribute("placeholder") or ""
        print(f"  input[{i}]: type={inp_type} name={inp_name} id={inp_id} placeholder={inp_placeholder}")

    buttons = await page.query_selector_all("button, input[type='submit'], a.btn, a.button")
    for i, btn in enumerate(buttons):
        tag = await btn.evaluate("el => el.tagName")
        text = await btn.inner_text() if tag != "INPUT" else await btn.get_attribute("value")
        btn_id = await btn.get_attribute("id") or ""
        btn_class = await btn.get_attribute("class") or ""
        print(f"  button[{i}]: tag={tag} text={text} id={btn_id} class={btn_class}")

    # ログイン試行
    if all([member_id, pin, auth]):
        print("\n  --- ログイン試行 ---")
        # 加入者番号フィールドを探す
        member_input = await page.query_selector("input[name*='member'], input[name*='kaiin'], input[id*='member'], input[id*='kaiin']")
        if not member_input:
            # 名前で見つからない場合、テキスト入力の最初のフィールド
            text_inputs = await page.query_selector_all("input[type='text'], input[type='tel'], input[type='number']")
            if text_inputs:
                member_input = text_inputs[0]

        pin_input = await page.query_selector("input[type='password'], input[name*='pin'], input[name*='pass']")

        auth_input = await page.query_selector("input[name*='auth'], input[name*='ninsho']")
        if not auth_input:
            # 3番目のテキスト入力を認証番号とする
            text_inputs = await page.query_selector_all("input[type='text'], input[type='tel'], input[type='number'], input[type='password']")
            if len(text_inputs) >= 3:
                auth_input = text_inputs[2]

        if member_input:
            await member_input.fill(member_id)
            print(f"  加入者番号入力: OK")
        else:
            print(f"  [警告] 加入者番号フィールドが見つかりません")

        if pin_input:
            await pin_input.fill(pin)
            print(f"  暗証番号入力: OK")
        else:
            print(f"  [警告] 暗証番号フィールドが見つかりません")

        if auth_input:
            await auth_input.fill(auth)
            print(f"  認証番号入力: OK")
        else:
            print(f"  [警告] 認証番号フィールドが見つかりません")

        await save_snapshot(page, "02_login_filled")

        # ログインボタンクリック
        login_btn = await page.query_selector("input[type='submit'], button[type='submit']")
        if not login_btn:
            login_btn = await page.query_selector("a.btn, a.button, button")
        if login_btn:
            await login_btn.click()
            await page.wait_for_load_state("networkidle", timeout=15000)
            await save_snapshot(page, "03_after_login")
            print(f"  ログインボタンクリック: OK")
            print(f"  現在URL: {page.url}")
        else:
            print(f"  [警告] ログインボタンが見つかりません")


async def step_venue(page):
    """開催場選択画面の探索"""
    print("\n=== Step: venue ===")
    await save_snapshot(page, "04_venue_page")

    # リンク・ボタンの探索
    links = await page.query_selector_all("a")
    print(f"  リンク数: {len(links)}")
    for i, link in enumerate(links[:30]):
        text = (await link.inner_text()).strip()
        href = await link.get_attribute("href") or ""
        if text:
            print(f"  link[{i}]: text={text[:30]} href={href[:60]}")


async def step_race(page):
    """レース選択画面の探索"""
    print("\n=== Step: race ===")
    await save_snapshot(page, "05_race_page")

    links = await page.query_selector_all("a")
    for i, link in enumerate(links[:30]):
        text = (await link.inner_text()).strip()
        href = await link.get_attribute("href") or ""
        if text:
            print(f"  link[{i}]: text={text[:30]} href={href[:60]}")


async def step_bet(page):
    """購入画面の探索"""
    print("\n=== Step: bet ===")
    await save_snapshot(page, "06_bet_page")

    # フォーム要素
    inputs = await page.query_selector_all("input, select")
    for i, inp in enumerate(inputs):
        tag = await inp.evaluate("el => el.tagName")
        inp_type = await inp.get_attribute("type") or ""
        inp_name = await inp.get_attribute("name") or ""
        inp_id = await inp.get_attribute("id") or ""
        print(f"  {tag}[{i}]: type={inp_type} name={inp_name} id={inp_id}")


async def main(step):
    async with async_playwright() as pw:
        browser, context, page = await create_browser(pw)

        try:
            if step in ('login', 'all'):
                await step_login(page)

            if step in ('venue', 'all'):
                if step == 'venue':
                    # venue単体の場合はまずログインが必要
                    await step_login(page)
                await step_venue(page)

            if step in ('race', 'all'):
                if step == 'race':
                    await step_login(page)
                    await step_venue(page)
                await step_race(page)

            if step in ('bet', 'all'):
                if step == 'bet':
                    await step_login(page)
                await step_bet(page)

            print(f"\n完了。スクリーンショット保存先: {SS_DIR}")
            print("headedモードで起動しています。ブラウザを手動操作して画面構造を確認できます。")
            print("Enterキーで終了...")
            await asyncio.get_event_loop().run_in_executor(None, input)

        finally:
            await browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='テレボート画面構造探索ツール')
    parser.add_argument('--step', choices=['login', 'venue', 'race', 'bet', 'all'],
                        default='login', help='探索するステップ')
    args = parser.parse_args()
    asyncio.run(main(args.step))
