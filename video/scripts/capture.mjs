// Film a recorded live-demo run frame by frame from the real page (scripts/live_page.html), in replay with
// turns revealed as the replay reaches them, then encode it at 20 fps (2x sim time).
//
//   python scripts/live.py --name live_184743 --no-browser --port 8790      # in the repo root, serves the session
//   node scripts/capture.mjs --url http://localhost:8790 --ep ep_002 --name page_ep002
//
// Output: public/footage/<name>.mp4 (frames in public/footage/<name>/ are kept for stills).
import {execFileSync} from 'node:child_process';
import {mkdirSync} from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

const arg = (k, d) => {
  const i = process.argv.indexOf(`--${k}`);
  return i > 0 ? process.argv[i + 1] : d;
};
const url = arg('url', 'http://localhost:8790');
const ep = arg('ep', 'ep_002');
const name = arg('name', `page_${ep.replace('_', '')}`);
const theme = arg('theme', 'light');
const dir = path.resolve('public/footage', name);
mkdirSync(dir, {recursive: true});

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME || '/usr/bin/google-chrome',
  headless: true,
  args: ['--no-sandbox', '--hide-scrollbars', '--force-device-scale-factor=1'],
  defaultViewport: {width: 1920, height: 1080},
});
const page = await browser.newPage();
await page.goto(`${url}/?ep=${ep}&reveal=1&theme=${theme}`, {waitUntil: 'networkidle0'});
await page.waitForFunction(() => typeof S !== 'undefined' && S.final, {timeout: 30000});
await page.evaluate(() => document.fonts.ready);
const n = await page.evaluate(() => S.view.n_frames);
const t0 = Date.now();
for (let k = 0; k < n; k++) {
  await page.evaluate((k) => __frame(k), k);
  await page.screenshot({path: path.join(dir, `${String(k).padStart(5, '0')}.jpg`), type: 'jpeg', quality: 90});
  if (k % 100 === 0) console.log(`${ep}: frame ${k}/${n} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}
await browser.close();
execFileSync('ffmpeg', ['-v', 'error', '-y', '-framerate', '20', '-i', path.join(dir, '%05d.jpg'),
  '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', `${dir}.mp4`]);
console.log(`${dir}.mp4 (${n} frames)`);
