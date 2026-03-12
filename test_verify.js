const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
    const page = await browser.newPage();

    // Desktop view
    await page.setViewport({ width: 1280, height: 800 });
    try { await page.goto('https://web-production-c977.up.railway.app', { waitUntil: 'networkidle2', timeout: 45000 }); } catch(e) {}
    await new Promise(r => setTimeout(r, 8000));
    await page.screenshot({ path: 'C:/Users/iwashita.AKGNET/Pictures/Screenshots/boatrace_verify_top.png', fullPage: false });

    // Click tab1 "本日の買い目" to ensure we're on it
    const tab1 = await page.$('[data-testid="stTab"]:first-child');
    if (tab1) await tab1.click();
    await new Promise(r => setTimeout(r, 2000));
    await page.screenshot({ path: 'C:/Users/iwashita.AKGNET/Pictures/Screenshots/boatrace_verify_tab1.png', fullPage: false });

    // Scroll down to see expanders (closed state)
    await page.evaluate(() => window.scrollBy(0, 800));
    await new Promise(r => setTimeout(r, 1000));
    await page.screenshot({ path: 'C:/Users/iwashita.AKGNET/Pictures/Screenshots/boatrace_verify_expanders.png', fullPage: false });

    // Click one expander to verify it opens
    const firstExpander = await page.$('[data-testid="stExpander"] summary');
    if (firstExpander) {
        await firstExpander.click();
        await new Promise(r => setTimeout(r, 1500));
        await page.screenshot({ path: 'C:/Users/iwashita.AKGNET/Pictures/Screenshots/boatrace_verify_expanded.png', fullPage: false });
    }

    // Scroll test (performance)
    await page.evaluate(() => window.scrollTo(0, 0));
    await new Promise(r => setTimeout(r, 500));
    const scrollStart = Date.now();
    for (let i = 0; i < 20; i++) {
        await page.evaluate(() => window.scrollBy(0, 300));
        await new Promise(r => setTimeout(r, 100));
    }
    const scrollTime = Date.now() - scrollStart;
    console.log(`Scroll: ${scrollTime}ms for 20 scrolls`);

    // Mobile view
    await page.setViewport({ width: 393, height: 852, isMobile: true, hasTouch: true });
    await page.reload({ waitUntil: 'networkidle2', timeout: 45000 }).catch(() => {});
    await new Promise(r => setTimeout(r, 8000));
    await page.screenshot({ path: 'C:/Users/iwashita.AKGNET/Pictures/Screenshots/boatrace_verify_mobile.png', fullPage: false });

    console.log('All screenshots saved');
    await browser.close();
})();
