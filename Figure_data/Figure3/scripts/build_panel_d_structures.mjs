import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const figureDir = path.resolve(scriptDir, "..");
const packageDir = path.resolve(figureDir, "..");
const dataDir = path.resolve(
  process.env.PANEL_D_DATA_DIR ||
    path.join(figureDir, "data", "panel_d_structures"),
);
const outDir = path.resolve(
  process.env.PANEL_D_OUT_DIR ||
    path.join(figureDir, "plots", "panel_d_structures_reproduced"),
);
const serverRoot = path.resolve(process.env.RENDER_SERVER_ROOT || packageDir);
const serverUrl = new URL(
  process.env.RENDER_SERVER_URL || "http://127.0.0.1:8765/",
);
const renderPage = path.resolve(
  process.env.RENDER_PAGE_PATH || path.join(scriptDir, "render_structure.html"),
);

const files = ["cluster6_sim.pdb", "cluster88_sim.pdb", "c6.cif", "c88.cif"];

const rotationPresets = {
  "cluster6_sim.pdb": { x: -12, y: 18, z: -10 },
  "c6.cif": { x: -10, y: 45, z: -12 },
};

const scalePresets = {
  "cluster6_sim.pdb": 0.1,
};

function serverPath(filePath) {
  const relative = path.relative(serverRoot, filePath);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`${filePath} is outside RENDER_SERVER_ROOT=${serverRoot}`);
  }
  return `/${relative.split(path.sep).map(encodeURIComponent).join("/")}`;
}

function degToRad(deg) {
  return (deg * Math.PI) / 180;
}

function multiplyQuat(a, b) {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

function quatFromAxisAngle(axis, degrees) {
  const half = degToRad(degrees) / 2;
  const s = Math.sin(half);
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(half)];
}

function quatFromEulerXYZ(xDeg, yDeg, zDeg) {
  const qx = quatFromAxisAngle([1, 0, 0], xDeg);
  const qy = quatFromAxisAngle([0, 1, 0], yDeg);
  const qz = quatFromAxisAngle([0, 0, 1], zDeg);
  return multiplyQuat(qz, multiplyQuat(qy, qx));
}

await fs.mkdir(outDir, { recursive: true });
await fs.access(renderPage);
for (const name of files) {
  await fs.access(path.join(dataDir, name));
}

const launchOptions = { headless: true };
if (process.env.CHROME_EXECUTABLE_PATH) {
  launchOptions.executablePath = process.env.CHROME_EXECUTABLE_PATH;
} else if (process.env.PLAYWRIGHT_BROWSER_CHANNEL) {
  launchOptions.channel = process.env.PLAYWRIGHT_BROWSER_CHANNEL;
}

const browser = await chromium.launch(launchOptions);
try {
  const page = await browser.newPage({
    viewport: { width: 2200, height: 2200 },
    deviceScaleFactor: 1,
  });

  page.on("console", (msg) => {
    console.log(`[browser:${msg.type()}] ${msg.text()}`);
  });

  page.on("pageerror", (err) => {
    console.log(`[pageerror] ${err.message}`);
  });

  for (const name of files) {
    const url = new URL(serverPath(renderPage), serverUrl);
    url.searchParams.set("file", serverPath(path.join(dataDir, name)));
    if (scalePresets[name]) {
      url.searchParams.set("scale", scalePresets[name]);
    }
    console.log(`Rendering ${name}`);
    await page.goto(url.href, { waitUntil: "networkidle", timeout: 120000 });
    await page.waitForFunction(
      () => window.renderStatus && window.renderStatus.reason !== "loading",
      null,
      { timeout: 120000 },
    );
    const status = await page.evaluate(() => window.renderStatus);
    console.log(`Render status for ${name}: ${JSON.stringify(status)}`);
    if (!status || status.ok !== true) {
      throw new Error(`Render failed for ${name}: ${JSON.stringify(status)}`);
    }
    const preset = rotationPresets[name];
    if (preset) {
      const quat = quatFromEulerXYZ(preset.x, preset.y, preset.z);
      await page.evaluate((q) => {
        window.stage.viewerControls.rotate(q);
        window.stage.viewer.requestRender();
      }, quat);
      await page.waitForTimeout(500);
    }
    await page.screenshot({
      path: path.join(outDir, `${path.parse(name).name}_unified.png`),
      type: "png",
      omitBackground: false,
    });
  }
} finally {
  await browser.close();
}

console.log(`input: ${dataDir}`);
console.log(`output: ${outDir}`);
