const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
    const page = await browser.newPage();
    await page.setViewport({ width: 393, height: 852, isMobile: true, hasTouch: true });

    try { await page.goto('https://web-production-c977.up.railway.app', { waitUntil: 'networkidle2', timeout: 45000 }); } catch(e) {}
    await new Promise(r => setTimeout(r, 10000));

    // Check deployment version - look for summary table
    const hasSummaryTable = await page.evaluate(() => {
        const text = document.body.innerText;
        return text.includes('会場別サマリー');
    });
    console.log('Has summary table (new deploy):', hasSummaryTable);

    // Check first expander DOM structure
    const expanderDOM = await page.evaluate(() => {
        const expander = document.querySelector('[data-testid="stExpander"]');
        if (!expander) return 'No expander found';

        const details = expander.querySelector('[data-testid="stExpanderDetails"]');
        if (!details) return 'No details element found';

        return {
            detailsHidden: details.hidden,
            detailsDisplay: window.getComputedStyle(details).display,
            detailsVisibility: window.getComputedStyle(details).visibility,
            detailsOuterHTML: details.outerHTML.substring(0, 500),
            expanderSummaryOpen: expander.querySelector('details')?.open,
            expanderOuterHTML: expander.outerHTML.substring(0, 800),
        };
    });
    console.log('Expander DOM:', JSON.stringify(expanderDOM, null, 2));

    // Count with different methods
    const counts = await page.evaluate(() => {
        const all = document.querySelectorAll('[data-testid="stExpander"]');
        let hiddenAttr = 0;
        let displayNone = 0;
        let detailsOpen = 0;
        let detailsClosed = 0;

        all.forEach(e => {
            const det = e.querySelector('[data-testid="stExpanderDetails"]');
            if (det && det.hidden) hiddenAttr++;
            if (det && window.getComputedStyle(det).display === 'none') displayNone++;

            const detailsEl = e.querySelector('details');
            if (detailsEl) {
                if (detailsEl.open) detailsOpen++;
                else detailsClosed++;
            }
        });

        return { total: all.length, hiddenAttr, displayNone, detailsOpen, detailsClosed };
    });
    console.log('Counts:', JSON.stringify(counts));

    // Check tab content - is tab1 even visible?
    const tabInfo = await page.evaluate(() => {
        const tabs = document.querySelectorAll('[data-testid="stTab"]');
        return { tabCount: tabs.length };
    });
    console.log('Tabs:', JSON.stringify(tabInfo));

    await browser.close();
})();
