#!/usr/bin/env node
// Copies the current TAS-Standalone and TAS-AdobeEdition builds from the update
// host onto the GitHub release of the engine version they pin.
//
// The update host (https://updates.nevermindnilas.dev) is what clients read. The
// GitHub release is the second copy: first downloads, clients that cannot reach
// the host, and installs that predate it. Those older Standalone builds read
// stable.yml from whatever release GitHub calls Latest, and older Adobe panels
// look there for TAS-<tag>-AdobeEdition-<os>.zip, so the release for an engine
// version has to carry the frontends pinned to it.
//
// For each frontend: read <product>/stable/manifest.json, find the release tagged
// v<manifest.engine> (drafts included), and make that release's frontend assets
// equal the manifest. New files go up first, then the feed files and a copy of the
// manifest (each swapped in by rename, so the old one is never missing), then stale
// files are removed. A published release never has a feed pointing at a missing
// installer.
//
// One Adobe file is never removed: TAS-<tag digits>-AdobeEdition-<os>.zip on its own
// release. A 3.1.0 panel can only update from the zip whose number equals the tag, so
// that build stays as the bridge even after a newer Adobe-only build is current.
//
//   GH_TOKEN=... node .github/scripts/sync-frontends.mjs [--dry-run] [--repo owner/name]
//
// No dependencies, Node 20 or newer.

import { createHash } from "node:crypto";
import { createReadStream, createWriteStream, openAsBlob } from "node:fs";
import { mkdtemp, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

const HOST = (process.env.CDN_PUBLIC_BASE ?? "https://updates.nevermindnilas.dev").replace(/\/+$/, "");
const args = process.argv.slice(2);
const DRY_RUN = args.includes("--dry-run");
const REPO = args.includes("--repo") ? args[args.indexOf("--repo") + 1] : process.env.GITHUB_REPOSITORY ?? "NevermindNilas/TheAnimeScripter";
const TOKEN = process.env.GH_TOKEN ?? process.env.GITHUB_TOKEN;
const CHANNEL = "stable";

/** Which release assets belong to each frontend. Everything else is left alone. */
const PRODUCTS = {
  standalone: {
    owns: (name) => /-Standalone-/i.test(name) || /^stable(-mac|-linux)?\.yml(\.new)?$/.test(name),
    manifestAsset: "TAS-Standalone-manifest.json",
    keeps: () => false,
  },
  adobe: {
    owns: (name) => /-AdobeEdition-/i.test(name),
    manifestAsset: "TAS-AdobeEdition-manifest.json",
    // The bridge build for panels that predate the update host (see the header).
    keeps: (name, tag) => new RegExp(`^TAS-${tag.replace(/\D/g, "")}-AdobeEdition-(windows|macos)\\.zip$`, "i").test(name),
  },
};

const notice = (message) => console.log(process.env.GITHUB_ACTIONS ? `::notice::${message}` : message);
const warning = (message) => console.log(process.env.GITHUB_ACTIONS ? `::warning::${message}` : `WARNING: ${message}`);

/** One GitHub API call. Network errors and 5xx are retried; `init()` rebuilds the request each time. */
async function github(path, init = {}) {
  const url = path.startsWith("https://") ? path : `https://api.github.com${path}`;
  const build = typeof init === "function" ? init : () => init;
  let last;
  for (let attempt = 1; attempt <= 4; attempt++) {
    const options = await build();
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          accept: "application/vnd.github+json", "x-github-api-version": "2022-11-28", "user-agent": "tas-sync-frontends",
          ...(TOKEN ? { authorization: `Bearer ${TOKEN}` } : {}), ...options.headers,
        },
      });
      if (response.ok) return response.status === 204 ? null : response.json();
      last = new Error(`${options.method ?? "GET"} ${url} -> ${response.status} ${(await response.text()).slice(0, 300)}`);
      if (response.status < 500 && response.status !== 429) throw Object.assign(last, { final: true });
    } catch (error) {
      if (error.final) throw error;
      last = error;
    }
    if (attempt < 4) await new Promise((r) => setTimeout(r, 3000 * attempt));
  }
  throw last;
}

/** Every release, drafts included when the token can see them, newest first. */
async function listReleases() {
  const all = [];
  for (let page = 1; page <= 10; page++) {
    const batch = await github(`/repos/${REPO}/releases?per_page=100&page=${page}`);
    all.push(...batch);
    if (batch.length < 100) break;
  }
  return all;
}

async function sha256Of(path) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

async function download(file, dir) {
  const target = join(dir, file.name);
  let last;
  for (let attempt = 1; attempt <= 4; attempt++) {
    try {
      const response = await fetch(file.url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      await pipeline(Readable.fromWeb(response.body), createWriteStream(target));
      const size = (await stat(target)).size;
      if (size !== file.size) throw new Error(`got ${size} bytes, manifest says ${file.size}`);
      const sum = await sha256Of(target);
      if (sum !== file.sha256) throw new Error(`sha256 ${sum} does not match the manifest`);
      return target;
    } catch (error) {
      last = error;
      if (attempt < 4) await new Promise((r) => setTimeout(r, 3000 * attempt));
    }
  }
  throw new Error(`could not fetch ${file.url}: ${last?.message ?? last}`);
}

/** Deletes a release asset. One that is already gone counts as deleted. */
async function deleteAsset(asset) {
  try { await github(`/repos/${REPO}/releases/assets/${asset.id}`, { method: "DELETE" }); }
  catch (error) { if (!/-> 404 /.test(error.message)) throw error; }
}

async function uploadAsset(release, name, path) {
  const base = release.upload_url.replace(/\{.*$/, "");
  // A failed earlier attempt can leave a half-uploaded asset under this name; clear it before each try.
  return github(`${base}?name=${encodeURIComponent(name)}`, async () => {
    const fresh = await github(`/repos/${REPO}/releases/${release.id}/assets?per_page=100`);
    for (const asset of fresh.filter((a) => a.name === name)) await deleteAsset(asset);
    return { method: "POST", body: await openAsBlob(path), headers: { "content-type": "application/octet-stream" } };
  });
}

/**
 * Replaces a small text asset (a feed or the manifest copy) without a moment where
 * it is missing for long: upload under a temporary name, drop the old one, rename.
 */
async function swapTextAsset(release, name, path, existing) {
  const staged = await uploadAsset(release, `${name}.new`, path);
  if (existing) await deleteAsset(existing);
  await github(`/repos/${REPO}/releases/assets/${staged.id}`, { method: "PATCH", body: JSON.stringify({ name }), headers: { "content-type": "application/json" } });
}

/** The text of a small release asset (a feed file). Follows GitHub's redirect to storage. */
async function assetText(asset) {
  const response = await fetch(asset.url, {
    headers: { accept: "application/octet-stream", "user-agent": "tas-sync-frontends", ...(TOKEN ? { authorization: `Bearer ${TOKEN}` } : {}) },
  });
  if (!response.ok) throw new Error(`GET ${asset.name} -> ${response.status}`);
  return response.text();
}

async function syncProduct(product, releases, latestTag) {
  const response = await fetch(`${HOST}/${product}/${CHANNEL}/manifest.json`, { cache: "no-store" });
  if (response.status === 404) { notice(`${product}: nothing published on the update host yet`); return; }
  if (!response.ok) throw new Error(`${product}: manifest fetch failed with HTTP ${response.status}`);
  const manifestText = await response.text();
  const manifest = JSON.parse(manifestText);
  if (manifest.schema !== 1 || manifest.product !== product || !manifest.engine || !Array.isArray(manifest.files)) {
    throw new Error(`${product}: manifest.json is not a schema 1 ${product} release`);
  }
  for (const file of manifest.files) {
    if (!PRODUCTS[product].owns(file.name)) throw new Error(`${product}: ${file.name} does not look like a ${product} file; refusing to sync`);
  }

  const tag = `v${manifest.engine}`;
  const release = releases.find((r) => r.tag_name === tag);
  if (!release) { notice(`${product} ${manifest.version} pins engine ${manifest.engine}, but there is no ${tag} release in ${REPO} yet`); return; }

  // After a rollback on the update host to a build pinning an older engine, a newer
  // release still carries the build that was rolled back. If that newer release is the
  // one GitHub calls Latest, installs that predate the update host keep being offered
  // the bad build, so the run fails until a human resolves it. Deleting the files is
  // the wrong fix (a Latest release without stable*.yml breaks their update checks).
  for (const newer of releases) {
    if (newer.draft || compareTags(newer.tag_name, tag) <= 0) continue;
    if (!newer.assets.some((a) => PRODUCTS[product].owns(a.name))) continue;
    const message = `${newer.tag_name} carries ${product} files, but the current ${product} build (${manifest.version}) pins engine ${manifest.engine}`;
    if (newer.tag_name === latestTag) {
      throw new Error(`${message}, and ${newer.tag_name} is the Latest release: installs older than the update host are still offered the build that was rolled back. Mark ${tag} as Latest (gh release edit ${tag} --latest), or publish a fixed ${product} build pinned to ${newer.tag_name.replace(/^v/, "")}. Do not delete the ${product} files from ${newer.tag_name}.`);
    }
    warning(`${message}; ${newer.tag_name} is not Latest, so nothing reads them. Remove them if that build was rolled back.`);
  }

  // Text assets: the feed files plus a copy of the manifest, which is what a client
  // falling back to GitHub reads (the release tag is the engine version, not the frontend's).
  const feeds = { ...(manifest.feeds ?? {}), [PRODUCTS[product].manifestAsset]: manifestText };
  const owned = release.assets.filter((a) => PRODUCTS[product].owns(a.name));
  const wantedNames = new Set([...manifest.files.map((f) => f.name), ...Object.keys(feeds)]);
  const present = new Map(owned.map((a) => [a.name, a]));

  const matches = (asset, file) =>
    asset?.state === "uploaded" && asset.size === file.size && (!asset.digest || asset.digest === `sha256:${file.sha256}`);
  const filesToUpload = manifest.files.filter((f) => !matches(present.get(f.name), f));
  const feedsToUpload = [];
  for (const [name, text] of Object.entries(feeds)) {
    const asset = present.get(name);
    if (!asset || asset.state !== "uploaded" || asset.size !== Buffer.byteLength(text) || (await assetText(asset)) !== text) feedsToUpload.push(name);
  }
  // A `<name>.new` left by an interrupted swap is cleared by the swap itself when that
  // name is replaced this run; listing it here too would delete it twice.
  const swappedThisRun = new Set(feedsToUpload.map((name) => `${name}.new`));
  const stale = owned.filter((a) => !wantedNames.has(a.name) && !swappedThisRun.has(a.name) && !PRODUCTS[product].keeps(a.name, tag));

  const label = `${product} ${manifest.version} -> ${tag}${release.draft ? " (draft)" : ""}`;
  if (!filesToUpload.length && !feedsToUpload.length && !stale.length) { console.log(`${label}: up to date`); return; }
  console.log(`${label}: upload ${[...filesToUpload.map((f) => f.name), ...feedsToUpload].join(", ") || "nothing"}; remove ${stale.map((a) => a.name).join(", ") || "nothing"}`);
  if (DRY_RUN) return;

  const dir = await mkdtemp(join(tmpdir(), `tas-sync-${product}-`));
  try {
    // Fetch and verify everything before touching the release.
    const paths = new Map();
    for (const file of filesToUpload) paths.set(file.name, await download(file, dir));
    for (const name of feedsToUpload) { const path = join(dir, name); await writeFile(path, feeds[name]); paths.set(name, path); }

    for (const file of filesToUpload) {
      // A release file's name carries its version, so an existing one under this name is a broken upload.
      await uploadAsset(release, file.name, paths.get(file.name));
      console.log(`  + ${file.name}`);
    }
    for (const name of feedsToUpload) {
      await swapTextAsset(release, name, paths.get(name), present.get(name));
      console.log(`  + ${name}`);
    }
    for (const asset of stale) { await deleteAsset(asset); console.log(`  - ${asset.name}`); }
    notice(`${label}: synced`);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

/** Orders vX.Y.Z tags; anything else sorts lowest. */
function compareTags(a, b) {
  const parse = (t) => /^v?(\d+)\.(\d+)\.(\d+)$/.exec(t)?.slice(1).map(Number);
  const pa = parse(a), pb = parse(b);
  if (!pa || !pb) return (pa ? 1 : 0) - (pb ? 1 : 0);
  for (let i = 0; i < 3; i++) if (pa[i] !== pb[i]) return pa[i] - pb[i];
  return 0;
}

if (!TOKEN && !DRY_RUN) { console.error("sync-frontends: GH_TOKEN is not set"); process.exit(1); }
const releases = await listReleases();
let failed = false;
// What GitHub itself calls Latest (it can be set by hand, so it is asked, not guessed).
const latestTag = await github(`/repos/${REPO}/releases/latest`).then((r) => r.tag_name, () => null);
for (const product of Object.keys(PRODUCTS)) {
  try { await syncProduct(product, releases, latestTag); }
  catch (error) { failed = true; console.error(process.env.GITHUB_ACTIONS ? `::error::${error.message}` : `ERROR: ${error.message}`); }
}

// Installs that predate the update host read the Standalone feed from the release GitHub
// marks Latest. A Latest without one means the draft was published before the frontends
// were attached; fail the run so the scheduled job reports it until it is fixed.
try {
  const latest = await github(`/repos/${REPO}/releases/latest`);
  const missing = ["stable.yml", "stable-mac.yml", "stable-linux.yml"].filter((n) => !latest.assets.some((a) => a.name === n && a.state === "uploaded"));
  if (missing.length) {
    failed = true;
    const message = `${latest.tag_name} is the Latest release but has no ${missing.join(", ")}: Standalone installs older than the update host cannot check for updates until a Standalone build pinned to ${latest.tag_name.replace(/^v/, "")} is published and synced`;
    console.error(process.env.GITHUB_ACTIONS ? `::error::${message}` : `ERROR: ${message}`);
  }
} catch (error) {
  failed = true;
  console.error(process.env.GITHUB_ACTIONS ? `::error::${error.message}` : `ERROR: ${error.message}`);
}
process.exit(failed ? 1 : 0);
