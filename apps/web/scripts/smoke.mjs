/**
 * Drives the three screens in a real browser against a running API.
 *
 * This is not a unit test and does not run in CI: it needs uvicorn, a database with an
 * ingested policy, and an OpenAI key, because the two answers it checks are the two
 * behaviours worth checking — a cited answer and an abstention. It exists because
 * "it compiles and the routes return 200" says nothing about whether the citation panel
 * renders the numbers it is for, and because it catches console errors that a build does
 * not. It found two on its first run.
 *
 *   uvicorn services.api.main:app --port 8000     # terminal 1
 *   npm run dev                                   # terminal 2
 *   npm run smoke                                 # terminal 3
 *
 * Uses the locally installed Chrome rather than downloading a browser.
 */

import { chromium } from "playwright-core";

const WEB = (process.env.SMOKE_WEB_URL ?? "http://localhost:3000").replace(/\/+$/, "");
const SHOTS = process.env.SMOKE_SHOT_DIR ?? null;
const ANSWERABLE = "What is the grace period for paying premiums?";
const UNANSWERABLE = "What is the premium for a 35 year old non-smoker?";

const failures = [];
function check(condition, message) {
  if (!condition) failures.push(message);
}

const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });

const consoleErrors = [];
page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
});
page.on("pageerror", (error) => consoleErrors.push(String(error.message)));

async function shoot(name) {
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: true });
}

// --- Library -------------------------------------------------------------------------
await page.goto(`${WEB}/`, { waitUntil: "networkidle" });
await page.waitForSelector("table", { timeout: 15000 });
const rows = await page.locator("tbody tr").count();
check(rows > 0, "the library showed no documents -- is anything ingested?");
await shoot("library");

// --- Upload --------------------------------------------------------------------------
await page.goto(`${WEB}/upload`, { waitUntil: "networkidle" });
check(await page.getByLabel("PDF").isVisible(), "the upload form has no file input");
await shoot("upload");

// --- Ask: a question the documents answer --------------------------------------------
await page.goto(`${WEB}/ask`, { waitUntil: "networkidle" });
await page.getByPlaceholder("What does this policy say about").fill(ANSWERABLE);
await page.getByRole("button", { name: "Ask", exact: true }).click();
await page.waitForSelector("text=Answer", { timeout: 90000 });
await page.waitForSelector("[data-testid='chunk-text']", { timeout: 10000 });

const first = page.locator("div.rounded-lg.border.p-4").first();
const evidence = await first.innerText();
check(/\bpp?\. \d/.test(evidence), "the citation card shows no page number");
check(/ranking score/.test(evidence), "the citation card is missing the ranking score");
check(/cosine/.test(evidence), "the citation card is missing the cosine");

// The point of showing both: under any hybrid mode they are different numbers on
// different scales. If they ever print as one number, the panel has stopped explaining
// the thing it exists to explain.
const numbers = [...evidence.matchAll(/^0\.\d{4}$/gm)].map((m) => Number(m[0]));
check(numbers.length >= 2, "the citation card did not render two scores");
check(
  numbers.length < 2 || numbers[0] !== numbers[1],
  `ranking score and cosine are identical (${numbers[0]}) -- expected different scales`,
);
await shoot("ask-answer");

// --- Ask: a question they do not answer ----------------------------------------------
await page.getByPlaceholder("What does this policy say about").fill(UNANSWERABLE);
await page.getByRole("button", { name: "Ask", exact: true }).click();
await page.waitForSelector("text=Declined to answer", { timeout: 90000 });
const abstention = await page.locator("text=Declined to answer").first().locator("xpath=../..").innerText();
check(abstention.trim().length > "Declined to answer".length + 10,
  "the abstention rendered without a reason");
await shoot("ask-abstain");

check(consoleErrors.length === 0, `console errors: ${consoleErrors.join(" | ")}`);
await browser.close();

if (failures.length > 0) {
  console.error("FAILED");
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}
console.log(`ok -- library (${rows} rows), cited answer, abstention, no console errors`);
