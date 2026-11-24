#!/usr/bin/env node
// Publishes a release to the TAS update host (https://updates.nevermindnilas.dev).
//
// CANONICAL COPY: TAS-Updates-Host/client/cdn-publish.mjs. Each repo vendors this
// file unchanged; fix bugs there and re-copy, do not fork it.
//
// No dependencies, Node 20 or newer. Auth is a per-product bearer token in
// CDN_PUBLISH_TOKEN (a GitHub Actions secret); the gateway decides what it may write.
//
//   node cdn-publish.mjs release --product adobe --channel stable --version 3.1.3 \
//        --engine 3.1.2 --file windows:zip:dist/a.zip --file macos:zip:dist/b.zip \
//        [--feed release/stable.yml ...] [--notes-file notes.md] [--dry-run]
//   node cdn-publish.mjs rollback --product adobe --channel stable --version 3.1.2
//   node cdn-publish.mjs status
//
// `release` uploads every file, then commits the manifest. Nothing is visible to
// users until the commit, and a published version can never be replaced: a rerun
// with identical files is a no-op, a rerun with different files is refused.
//
// Exit codes: 0 published and verified; 1 nothing was published (or it is unknown);
// 2 the release IS published but checking it afterwards failed. On 2, do not rerun:
// roll back and publish a new version.

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { openAsBlob } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { basename } from "node:path";

const GATEWAY = (process.env.CDN_PUBLISH_URL ?? "https://tas-updates-gateway.nilascontact.workers.dev").replace(/\/+$/, "");
const PUBLIC_BASE = (process.env.CDN_PUBLIC_BASE ?? "https://updates.nevermindnilas.dev").replace(/\/+$/, "");
const PART_BYTES = 48 * 1024 * 1024;      // under the gateway's 100 MB request limit
const SINGLE_PUT_MAX = 48 * 1024 * 1024;
// Enough attempts to outlast the gateway's longest lock lease (two minutes) when another
// publish or a prune is running.
const ATTEMPTS = 8;
/** Exit code for "the release is committed, but checking it afterwards failed". */
const EXIT_LIVE_BUT_UNVERIFIED = 2;
/** An upload nobody committed is deleted by the host after 48 hours; one older than this is sent again. */
const REUPLOAD_AFTER_MS = 24 * 3600 * 1000;

const die = (message) => { console.error(`cdn-publish: ${message}`); process.exit(1); };

function parseArgs(argv) {
  const [command, ...rest] = argv;
  const opts = { file: [], feed: [] };
  for (let i = 0; i < rest.length; i++) {
    const arg = rest[i];
    if (!arg.startsWith("--")) die(`unexpected argument ${arg}`);
    const name = arg.slice(2);
    if (name === "dry-run") { opts.dryRun = true; continue; }
    const value = rest[++i];
    if (value === undefined) die(`--${name} needs a value`);
    if (name === "file" || name === "feed") opts[name].push(value); else opts[name] = value;
  }
  return { command, opts };
}

async function sha256Of(path) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

async function call(method, path, { body, headers = {}, query = {} } = {}) {
  const token = process.env.CDN_PUBLISH_TOKEN;
  if (!token) die("CDN_PUBLISH_TOKEN is not set");
  const url = new URL(GATEWAY + path);
  for (const [k, v] of Object.entries(query)) url.searchParams.set(k, String(v));
  let last;
  for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
    try {
      const response = await fetch(url, { method, body, headers: { authorization: `Bearer ${token}`, ...headers } });
      const text = await response.text();
      let data; try { data = JSON.parse(text); } catch { data = { ok: false, error: text.slice(0, 300) }; }
      if (response.ok) return data;
      // 4xx is the gateway saying no: retrying cannot change the answer. 503 means
      // another publish or a prune holds the lock, which a retry does fix.
      if (response.status < 500 && response.status !== 429) {
        throw Object.assign(new Error(`${method} ${path} -> ${response.status}: ${data.error}${data.details ? "\n  - " + data.details.join("\n  - ") : ""}`), { final: true });
      }
      last = new Error(`${response.status}: ${data.error}`);
    } catch (error) {
      if (error.final) throw error;
      last = error;
    }
    if (attempt < ATTEMPTS) await new Promise((r) => setTimeout(r, 3000 * attempt));
  }
  throw new Error(`${method} ${path} failed after ${ATTEMPTS} attempts: ${last?.message ?? last}`);
}

async function upload(key, path, size, sha256) {
  const info = await call("GET", "/v1/info", { query: { key } });
  const fresh = info.uploaded && Date.now() - Date.parse(info.uploaded) < REUPLOAD_AFTER_MS;
  if (info.exists && info.size === size && info.sha256 === sha256 && (info.committed || fresh)) { console.log(`  = ${key} (already uploaded)`); return; }
  if (info.status === "frozen") throw new Error(`${key}: this channel's manifest.json or index.json is missing or unreadable on the host; it has to be repaired before anything can be published to it`);
  if (info.status === "retired") throw new Error(`${key} was used by a release that has since been pruned; file names can never be reused, bump the version`);
  if (info.committed) throw new Error(`${key} belongs to a published release with different contents; bump the version`);
  const blob = await openAsBlob(path);
  const headers = { "x-sha256": sha256 };
  if (size <= SINGLE_PUT_MAX) {
    await call("PUT", "/v1/object", { query: { key }, body: blob, headers });
  } else {
    const { uploadId } = await call("POST", "/v1/upload/create", { query: { key }, headers });
    const parts = [];
    try {
      for (let offset = 0, n = 1; offset < size; offset += PART_BYTES, n++) {
        const part = await call("PUT", "/v1/upload/part", { query: { key, uploadId, partNumber: n }, body: blob.slice(offset, Math.min(offset + PART_BYTES, size)) });
        parts.push({ partNumber: part.partNumber, etag: part.etag });
        process.stdout.write(`\r  ^ ${key} ${Math.min(100, Math.round(((offset + PART_BYTES) / size) * 100))}%`);
      }
      process.stdout.write("\n");
      await call("POST", "/v1/upload/complete", { query: { key, uploadId }, body: JSON.stringify({ parts }), headers: { "content-type": "application/json" } });
    } catch (error) {
      process.stdout.write("\n");
      await fetch(`${GATEWAY}/v1/upload/abort?key=${encodeURIComponent(key)}&uploadId=${encodeURIComponent(uploadId)}`, { method: "POST", headers: { authorization: `Bearer ${process.env.CDN_PUBLISH_TOKEN}` } }).catch(() => {});
      throw error;
    }
  }
  const after = await call("GET", "/v1/info", { query: { key } });
  if (!after.exists || after.size !== size || after.sha256 !== sha256) throw new Error(`${key} on the host does not match the local file after upload (size ${after.size}, expected ${size})`);
  console.log(`  + ${key} (${(size / 1048576).toFixed(1)} MB)`);
}

async function release(opts) {
  for (const required of ["product", "channel", "version"]) if (!opts[required]) die(`--${required} is required`);
  if (!opts.file.length) die("at least one --file os:kind:path is required");
  const group = `${opts.product}/${opts.channel}`;

  const files = [];
  for (const spec of opts.file) {
    // os:kind:path, split on the first two colons only so C:\paths survive.
    const first = spec.indexOf(":"), second = spec.indexOf(":", first + 1);
    if (first < 0 || second < 0) die(`--file must be os:kind:path, got ${spec}`);
    const path = spec.slice(second + 1);
    const info = await stat(path).catch(() => { throw new Error(`file not found: ${path}`); });
    files.push({ name: basename(path), os: spec.slice(0, first), kind: spec.slice(first + 1, second), size: info.size, sha256: await sha256Of(path), path });
  }
  const feeds = {};
  for (const path of opts.feed) feeds[basename(path)] = await readFile(path, "utf8");
  const manifest = {
    schema: 1, product: opts.product, channel: opts.channel, version: opts.version,
    ...(opts.engine ? { engine: opts.engine } : {}),
    ...(opts["notes-file"] ? { notes: (await readFile(opts["notes-file"], "utf8")).slice(0, 20000) } : {}),
    files: files.map(({ path, ...f }) => f),
    ...(Object.keys(feeds).length ? { feeds } : {}),
  };

  console.log(`Publishing ${group} ${opts.version}${opts.engine ? ` (engine ${opts.engine})` : ""}`);
  if (opts.dryRun) { console.log(JSON.stringify(manifest, null, 2)); console.log("dry run: nothing uploaded"); return; }
  for (const f of files) await upload(`${group}/${f.name}`, f.path, f.size, f.sha256);
  const result = await call("POST", "/v1/commit", { body: JSON.stringify(manifest), headers: { "content-type": "application/json" } });
  console.log(result.alreadyPublished ? `  = ${opts.version} was already published` : `  committed ${opts.version}`);

  // From here on the release is live. Anything that goes wrong is reported with its own
  // exit code, because rerunning cannot fix it and "nothing was published" would be a lie.
  try {
    // Read it back the way a client would: the public index, then every file at its real
    // URL, hashed. This is the copy users download, edge cache included.
    const published = (await publicIndex(group)).releases?.find((m) => m.version === opts.version);
    if (!published) throw new Error(`${opts.version} is missing from ${PUBLIC_BASE}/${group}/index.json after commit`);
    for (const f of published.files) await verifyPublic(f);
    console.log(`  verified ${published.files.length} files at ${PUBLIC_BASE}/${group}/${result.current ? " (now current)" : " (not current: a rollback or a newer release is in effect)"}`);
  } catch (error) {
    const first = !result.alreadyPublished && group.startsWith("engine/");
    console.error(`cdn-publish: ${group} ${opts.version} IS PUBLISHED, but checking it failed: ${error?.message ?? error}`);
    console.error(first
      ? "cdn-publish: do not rerun with rebuilt files. If the bytes are wrong, publish a new engine version; frontends cannot pin this one safely."
      : "cdn-publish: do not rerun with rebuilt files. If the bytes are wrong, roll back (cdn-publish.mjs rollback) and publish a new version.");
    process.exit(EXIT_LIVE_BUT_UNVERIFIED);
  }

  // Retention is applied by the host; asking for it here keeps storage tidy right after
  // a release. It also runs daily on its own, so a failure is only worth a note.
  try {
    const pruned = await call("POST", "/v1/prune", { query: { dry: "0" } });
    if (pruned.deleted?.length) console.log(`  pruned ${pruned.deleted.length} old files`);
  } catch (error) { console.log(`  note: prune did not run (${String(error?.message ?? error).split("\n")[0]}); the host prunes daily`); }
}

/** The channel's public index.json, with errors that say what was actually returned. */
async function publicIndex(group) {
  const url = `${PUBLIC_BASE}/${group}/index.json`;
  let last;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      const text = await response.text();
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      try { return JSON.parse(text); } catch { throw new Error(`not JSON (starts with ${JSON.stringify(text.slice(0, 60))})`); }
    } catch (error) { last = error; if (attempt < 3) await new Promise((r) => setTimeout(r, 3000 * attempt)); }
  }
  throw new Error(`${url}: ${last?.message ?? last}`);
}

/** Downloads a published file from its public URL and checks size and sha256. */
async function verifyPublic(file) {
  let last;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const response = await fetch(file.url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const hash = createHash("sha256");
      let size = 0;
      for await (const chunk of response.body) { hash.update(chunk); size += chunk.length; }
      const sum = hash.digest("hex");
      // Wrong bytes are not a transient fault: the published name serves something else.
      if (size === file.size && sum !== file.sha256) throw Object.assign(new Error(`serves different bytes (sha256 ${sum})`), { final: true });
      if (size !== file.size) throw new Error(`served ${size} bytes, expected ${file.size}`);
      return;
    } catch (error) {
      last = error;
      if (error.final) break;
      if (attempt < 3) await new Promise((r) => setTimeout(r, 5000 * attempt));
    }
  }
  throw new Error(`${file.url} failed verification: ${last?.message ?? last}`);
}

const { command, opts } = parseArgs(process.argv.slice(2));
try { await main(); } catch (error) { die(error?.message ?? String(error)); }

async function main() {
if (command === "release") await release(opts);
else if (command === "rollback") {
  for (const required of ["product", "channel", "version"]) if (!opts[required]) die(`--${required} is required`);
  console.log(await call("POST", "/v1/rollback", { body: JSON.stringify({ product: opts.product, channel: opts.channel, version: opts.version }), headers: { "content-type": "application/json" } }));
} else if (command === "status") console.log(JSON.stringify(await call("GET", "/v1/status"), null, 2));
else if (command === "prune") console.log(JSON.stringify(await call("POST", "/v1/prune", { query: { dry: opts.apply === "yes" ? "0" : "1" } }), null, 2));
else die("usage: cdn-publish.mjs release|rollback|status|prune [options]");
}
