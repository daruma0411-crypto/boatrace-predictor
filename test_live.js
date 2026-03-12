const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
    const page = await browser.newPage();
    await page.setViewport({ width: 393, height: 852, isMobile: true, hasTouch: true });

    try { await page.goto('https://web-production-c977.up.railway.app', { waitUntil: 'networkidle2', timeout: 45000 }); } catch(e) {}
    await new Promise(r => setTimeout(r, 8000));

    // Error check
    const errors = await page.evaluate(() => {
        return Array.from(document.querySelectorAll('[data-testid="stException"]')).map(e => e.textContent.substring(0,200)).join('\n');
    });
    console.log(errors ? 'ERRORS: ' + errors : 'No errors');

    // Count expanders and check if any are expanded
    const expanderInfo = await page.evaluate(() => {
        const expanders = document.querySelectorAll('[data-testid="stExpander"]');
        let expandedCount = 0;
        expanders.forEach(e => {
            if (e.querySelector('[data-testid="stExpanderDetails"]:not([hidden])')) expandedCount++;
        });
        return { total: expanders.length, expanded: expandedCount };
    });
    console.log(`Expanders: ${expanderInfo.total} total, ${expanderInfo.expanded} expanded`);

    // Scroll test
    const scrollStart = Date.now();
    for (let i = 0; i < 10; i++) {
        await page.evaluate(() => window.scrollBy(0, 500));
        await new Promise(r => setTimeout(r, 200));
    }
    const scrollTime = Date.now() - scrollStart;
    console.log(`Scroll test: ${scrollTime}ms for 10 scrolls (should be <5000ms)`);

    // Screenshot at bottom
    await page.screenshot({ path: 'C:/Users/iwashita.AKGNET/Pictures/Screenshots/boatrace_mobile_scroll.png' });
    // Screenshot at top
    await page.evaluate(() => window.scrollTo(0, 0));
    await new Promise(r => setTimeout(r, 500));
    await page.screenshot({ path: 'C:/Users/iwashita.AKGNET/Pictures/Screenshots/boatrace_mobile_top.png' });

    console.log('Screenshots saved');
    await browser.close();
})();
