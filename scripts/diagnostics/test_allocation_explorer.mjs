// Browser-level checks using installed Chrome and Node; no package downloads.
import {spawn} from 'node:child_process';
import {mkdtemp, writeFile, rm, mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

if (process.argv.length !== 4) throw new Error('Usage: node test_allocation_explorer.mjs INPUT.html EVIDENCE_DIRECTORY');
const input = resolve(process.argv[2]);
const output = resolve(process.argv[3]);
await mkdir(output, {recursive: true});
const profile = await mkdtemp(join(tmpdir(), 'dndsr-explorer-check-'));
const chrome = spawn('google-chrome', ['--headless=new', '--no-sandbox', '--disable-gpu', '--no-first-run',
    '--remote-allow-origins=*', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank']);
let socket;
const pending = new Map();
let nextId = 1, session;
const report = {checks: []};
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
function check(condition, label) {
    if (!condition) throw new Error(label);
    report.checks.push(label);
}
function call(method, params = {}, targetSession = session) {
    const id = nextId++;
    return new Promise((resolve, reject) => {
        pending.set(id, {resolve, reject});
        socket.send(JSON.stringify({id, method, params, ...(targetSession ? {sessionId: targetSession} : {})}));
    });
}
async function evaluate(expression) {
    const reply = await call('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
    if (reply.exceptionDetails) throw new Error(JSON.stringify(reply.exceptionDetails));
    return reply.result.value;
}
async function waitFor(expression) {
    for (let count = 0; count < 100; ++count) {
        if (await evaluate(expression)) return;
        await pause(20);
    }
    throw new Error(`Timed out: ${expression}`);
}
try {
    const address = await new Promise((resolve, reject) => {
        let output = '';
        chrome.stderr.on('data', chunk => {
            output += chunk;
            const found = output.match(/DevTools listening on (ws:\/\/\S+)/);
            if (found) resolve(found[1]);
        });
        chrome.once('error', reject);
        chrome.once('exit', code => reject(new Error(`Chrome exited ${code}`)));
    });
    socket = new WebSocket(address);
    await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
    socket.onmessage = event => {
        const result = JSON.parse(event.data);
        if (!result.id) return;
        const promise = pending.get(result.id);
        pending.delete(result.id);
        if (result.error) promise.reject(new Error(JSON.stringify(result.error)));
        else promise.resolve(result.result);
    };
    const target = await call('Target.createTarget', {url: 'about:blank'}, null);
    session = (await call('Target.attachToTarget', {targetId: target.targetId, flatten: true}, null)).sessionId;
    await call('Page.enable');
    await call('Emulation.setDeviceMetricsOverride', {width: 1024, height: 900, deviceScaleFactor: 1, mobile: false});
    await call('Page.navigate', {url: pathToFileURL(input).href});
    await waitFor(`document.querySelector('.payload')&&document.querySelector('.tile')`);
    check(await evaluate(`(()=>{const data=JSON.parse(document.querySelector('.payload').textContent);const snapshot=data.snapshots[0];const memory=document.querySelector('.device').value;const total=snapshot.allocations.filter(r=>memory==='all'||r.device===memory).reduce((sum,r)=>sum+r.bytes,0);return !window.openai&&document.querySelector('.count').textContent.includes(new Intl.NumberFormat().format(total))})()`), 'standalone file works without conversation APIs');
    await call('Page.addScriptToEvaluateOnNewDocument', {source: `window.__savedStates=[];window.openai={setWidgetState(s){window.__savedStates.push(s);return Promise.resolve()}};`});
    await call('Page.navigate', {url: pathToFileURL(input).href});
    await waitFor(`document.querySelector('.payload')&&document.querySelector('.tile')`);
    await evaluate(`(()=>{const data=JSON.parse(document.querySelector('.payload').textContent);if(data.snapshots.length<2)throw new Error('interaction comparison needs at least two snapshots');const snapshot=data.snapshots[1];window.dispatchEvent(new CustomEvent('openai:set_globals',{detail:{globals:{widgetState:{modelContent:{setup:snapshot.label,device:'cuda:0',path:['CFV']}}}}}));})()`);
    await waitFor(`document.querySelector('.crumbs [aria-current]')?.textContent==='CFV'`);
    check(await evaluate(`document.querySelector('.setup').value==='1'&&document.querySelector('.device').value==='cuda:0'`), 'saved setup, device and path restore');
    await evaluate(`document.querySelector('.crumbs').querySelectorAll('button')[1].click()`);
    await waitFor(`document.querySelector('.crumbs [aria-current]')?.textContent==='All allocations'`);
    const positions = await evaluate(`Array.from(document.querySelectorAll('.tile')).slice(0,2).map(e=>{const r=e.getBoundingClientRect();return {x:r.x+Math.min(r.width/2,80),y:r.y+60,name:e.querySelector('.tile-name').textContent}})`);
    await call('Input.dispatchMouseEvent', {type: 'mouseMoved', x: positions[0].x, y: positions[0].y});
    await pause(40);
    check(await evaluate(`document.querySelector('.offline-tooltip').matches(':popover-open')&&document.querySelector('.offline-tooltip').textContent.includes('bytes')&&document.querySelector('.inspect-path').textContent.includes(${JSON.stringify(positions[0].name)})`), 'immediate hover preview and standalone tooltip');
    await call('Input.dispatchMouseEvent', {type: 'mouseMoved', x: positions[1].x, y: positions[1].y});
    await pause(30);
    check(await evaluate(`document.querySelector('.offline-tooltip').textContent.includes(${JSON.stringify(positions[1].name)})`), 'tooltip switches directly between tiles');
    await evaluate(`window.__tile=document.querySelector('.tile');document.querySelector('.details').open=true`);
    await pause(40);
    check(await evaluate(`window.__tile===document.querySelector('.tile')&&document.querySelectorAll('.records button').length>1`), 'detail expansion keeps tile identity and hover state');
    await call('Input.dispatchMouseEvent', {type: 'mouseMoved', x: positions[0].x, y: positions[0].y});
    await call('Input.dispatchMouseEvent', {type: 'mousePressed', x: positions[0].x, y: positions[0].y, button: 'left', clickCount: 1});
    await call('Input.dispatchMouseEvent', {type: 'mouseReleased', x: positions[0].x, y: positions[0].y, button: 'left', clickCount: 1});
    await waitFor(`document.querySelector('.crumbs [aria-current]')?.textContent===${JSON.stringify(positions[0].name)}`);
    check(await evaluate(`!document.querySelector('.offline-tooltip').matches(':popover-open')`), 'navigation retires tooltip without a stale overlay');
    await evaluate(`document.querySelector('.setup').value='0';document.querySelector('.setup').dispatchEvent(new Event('change'))`);
    await pause(40);
    check(await evaluate(`document.querySelector('.crumbs [aria-current]').textContent===${JSON.stringify(positions[0].name)}`), 'setup comparison preserves the matching group path');
    await pause(160);
    await evaluate(`document.querySelector('.crumbs button').click()`);
    await waitFor(`document.querySelector('.crumbs [aria-current]')?.textContent==='All allocations'`);
    await pause(160);
    check(await evaluate(`(()=>{const old=window.__savedStates.at(-2);window.dispatchEvent(new CustomEvent('openai:set_globals',{detail:{globals:{widgetState:old}}}));return true})()`), 'host persistence echo dispatched');
    await pause(40);
    check(await evaluate(`document.querySelector('.crumbs [aria-current]').textContent==='All allocations'`), 'old persistence echoes do not rewind navigation');
    const timings = await evaluate(`(async()=>{const times=[];for(let i=0;i<24;++i){const up=document.querySelector('.crumbs button');const target=up.disabled?document.querySelector('.tile'):up;const t=performance.now();target.click();await new Promise(requestAnimationFrame);times.push(performance.now()-t)}return times})()`);
    report.navigation_ms = {samples: timings.length, median: [...timings].sort((a,b)=>a-b)[12], maximum: Math.max(...timings)};
    check(report.navigation_ms.median < 50 && report.navigation_ms.maximum < 200, 'navigation stays within the interactive frame budget');
    await evaluate(`(async()=>{for(let n=0;n<24&&!document.querySelector('section').classList.contains('is-leaf');n++){document.querySelector('.tile').click();await new Promise(requestAnimationFrame)}})()`);
    check(await evaluate(`document.querySelector('section').classList.contains('is-leaf')&&document.querySelector('.details').open&&document.querySelector('.records').textContent.includes('Identity')`), 'leaf selection exposes exact allocation metadata');
    await call('Input.dispatchKeyEvent', {type: 'keyDown', key: 'ArrowLeft', code: 'ArrowLeft'});
    await call('Input.dispatchKeyEvent', {type: 'keyUp', key: 'ArrowLeft', code: 'ArrowLeft'});
    await pause(40);
    check(await evaluate(`!document.querySelector('section').classList.contains('is-leaf')`), 'keyboard Left navigates out of an allocation leaf');
    await evaluate(`document.querySelector('.crumbs').querySelectorAll('button')[1].click();document.querySelector('.details').open=false`);
    await pause(40);
    const screen = await call('Page.captureScreenshot', {format: 'png'});
    await writeFile(join(output, 'interaction-wide.png'), Buffer.from(screen.data, 'base64'));
    await call('Emulation.setDeviceMetricsOverride', {width: 320, height: 850, deviceScaleFactor: 1, mobile: false});
    await pause(80);
    report.narrow = await evaluate(`({width:innerWidth,scroll:document.documentElement.scrollWidth,overflow:Array.from(document.querySelectorAll('body *')).filter(e=>e.getBoundingClientRect().right>321).slice(0,12).map(e=>({class:e.className,right:e.getBoundingClientRect().right,width:e.getBoundingClientRect().width})),map:document.querySelector('.map').getBoundingClientRect().right,tiles:Array.from(document.querySelectorAll('.tile')).map(e=>e.getBoundingClientRect().right)})`);
    const narrow = await call('Page.captureScreenshot', {format: 'png'});
    await writeFile(join(output, 'interaction-narrow.png'), Buffer.from(narrow.data, 'base64'));
    if(report.narrow.scroll>320) console.log(JSON.stringify(report.narrow));
    check(report.narrow.scroll<=320&&report.narrow.tiles.every(right=>right<=report.narrow.map+1), '320px layout has no horizontal overflow');
    report.status = 'passed';
    console.log(JSON.stringify(report, null, 2));
    await writeFile(join(output, 'interaction-checks.json'), JSON.stringify(report, null, 2) + '\n');
} finally {
    if (socket?.readyState === WebSocket.OPEN) { await call('Browser.close', {}, null).catch(() => {}); socket.close(); }
    chrome.kill('SIGTERM');
    await pause(300);
    await rm(profile, {recursive: true, force: true}).catch(() => {});
}
