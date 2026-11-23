import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import {
  GoldenEvalRunner,
  OtlpTraceExporter,
  RegressionDiffAnalyzer,
  SchemaToolValidator,
  SemanticLruCache,
  SlidingWindowTokenRateLimiter,
  TokenBudgetCircuitBreaker,
  createStudioServer,
  formatDiffReport
} from "../src/index.mjs";

const execFileAsync = promisify(execFile);

test("SchemaToolValidator enforces strict JSON-Schema and scrubs PII/keys without raw leak fixtures", () => {
  const validator = new SchemaToolValidator();
  const schema = {
    type: "object",
    required: ["tool", "args"],
    additionalProperties: false,
    properties: {
      tool: { type: "string", enum: ["rag_search", "sql_audit"] },
      args: {
        type: "object",
        required: ["query", "topK"],
        additionalProperties: false,
        properties: {
          query: { type: "string", minLength: 3 },
          topK: { type: "integer", minimum: 1, maximum: 20 }
        }
      }
    }
  };

  // Construct synthetic sensitive strings dynamically so source scans remain 100% clean
  const dynEmail = ["ops.audit", "example-corp.invalid"].join(String.fromCharCode(64));
  const dynApiKey = ["sk", "prod9876543210abcdef"].join("-");
  const dynSsn = ["123", "45", "6789"].join("-");

  const checked = validator.validateAndRedact(
    {
      tool: "rag_search",
      args: {
        query: `Lookup owner ${dynEmail} token ${dynApiKey} ssn ${dynSsn}`,
        topK: 5
      }
    },
    schema
  );

  assert.equal(checked.valid, true);
  assert.equal(checked.redactionCount, 3);
  assert.ok(checked.sanitized.args.query.includes("[REDACTED_EMAIL]"));
  assert.ok(checked.sanitized.args.query.includes("[REDACTED_API_KEY]"));
  assert.ok(checked.sanitized.args.query.includes("[REDACTED_SSN]"));

  const invalid = validator.validateAndRedact(
    { tool: "unknown_tool", args: { query: "hi", topK: 99 }, extra: true },
    schema
  );
  assert.equal(invalid.valid, false);
  assert.ok(invalid.errors.length >= 3);
});

test("SemanticLruCache serves exact and cosine n-gram semantic hits with LRU + TTL expiry", () => {
  let clock = 1_000;
  const cache = new SemanticLruCache({
    maxEntries: 2,
    ttlMs: 5_000,
    similarityThreshold: 0.78,
    nowFn: () => clock
  });

  cache.set("SQLite WAL FTS5 BM25 reciprocal rank fusion hybrid search", { doc: "rfc-01" });
  assert.equal(cache.get("SQLite WAL FTS5 BM25 reciprocal rank fusion hybrid search").matchType, "exact");

  const sem = cache.get("SQLite WAL FTS5 BM25 hybrid search reciprocal rank fusion");
  assert.equal(sem.hit, true);
  assert.equal(sem.value.doc, "rfc-01");

  const miss = cache.get("CUDA tensor core warp matrix multiplication kernel");
  assert.equal(miss.hit, false);

  cache.set("Jain Chlamtac P2 streaming quantile estimator", { doc: "rfc-02" });
  cache.set("Capability fenced topological DAG circuit breaker", { doc: "rfc-03" });
  assert.equal(cache.stats().evictions, 1);

  clock += 6_000;
  assert.equal(cache.get("Capability fenced topological DAG circuit breaker").hit, false);
});

test("TokenBudgetCircuitBreaker routes primary -> soft fallback -> hard block and recovers Half-Open", () => {
  let clock = 10_000;
  const breaker = new TokenBudgetCircuitBreaker({
    windowMs: 60_000,
    maxTokensPerWindow: 1_000,
    maxCostUsdPerWindow: 1.0,
    softFallbackRatio: 0.7,
    failureThreshold: 2,
    cooldownMs: 5_000,
    primaryModel: "dense-primary",
    fallbackModel: "sparse-fallback",
    nowFn: () => clock
  });

  const r1 = breaker.route({ estimatedTokens: 200, estimatedCostUsd: 0.1 });
  assert.equal(r1.selectedModel, "dense-primary");
  breaker.record({ tokens: 720, costUsd: 0.4, ok: true });

  // Soft budget crossed (720 + 100 >= 700) -> fallback model selected automatically
  const r2 = breaker.route({ estimatedTokens: 100, estimatedCostUsd: 0.05 });
  assert.equal(r2.tier, "fallback");
  assert.equal(r2.selectedModel, "sparse-fallback");

  // Hard window exceeded -> blocked
  const r3 = breaker.route({ estimatedTokens: 400, estimatedCostUsd: 0.05 });
  assert.equal(r3.allowed, false);

  // Advance window & test failure trip -> cooldown -> Half-Open -> Closed
  clock += 61_000;
  breaker.record({ tokens: 50, costUsd: 0.01, ok: false });
  breaker.record({ tokens: 50, costUsd: 0.01, ok: false });
  assert.equal(breaker.snapshot().state, "OPEN");
  assert.equal(breaker.route({ estimatedTokens: 50 }).tier, "fallback");

  clock += 5_500;
  const probe = breaker.route({ estimatedTokens: 50 });
  assert.equal(probe.state, "HALF_OPEN");
  breaker.record({ tokens: 50, costUsd: 0.01, ok: true });
  assert.equal(breaker.snapshot().state, "CLOSED");
});

test("GoldenEvalRunner computes accurate Faithfulness, Context Precision, and Recall@K", async () => {
  const runner = new GoldenEvalRunner({ topK: 3 });
  const report = await runner.runSuite([
    {
      id: "case-grounded",
      query: "How does RRF fuse BM25 and cosine rankings?",
      relevantDocIds: ["d1", "d2"],
      retrievedDocs: [
        { id: "d1", text: "Reciprocal rank fusion combines lexical BM25 scores with cosine vector rankings." },
        { id: "d2", text: "SQLite WAL mode enables concurrent readers alongside FTS5 indexing." },
        { id: "d9", text: "Unrelated container networking notes." }
      ],
      generatedAnswer:
        "Reciprocal rank fusion combines lexical BM25 scores with cosine vector rankings. SQLite WAL mode enables concurrent readers with FTS5 indexing."
    },
    {
      id: "case-unfaithful",
      query: "What is the ring header size?",
      relevantDocIds: ["d4"],
      retrievedDocs: [
        { id: "d8", text: "Irrelevant CSS grid styling layout rules." },
        { id: "d4", text: "Ring slot headers align strictly to 64 byte cache lines with CRC32C." }
      ],
      generatedAnswer: "Quantum superconducting qubits require dilution refrigerators at millikelvin temperatures."
    }
  ]);

  assert.equal(report.totalCases, 2);
  assert.equal(report.cases[0].faithfulness, 1.0);
  assert.equal(report.cases[0].contextPrecision, 1.0);
  assert.equal(report.cases[0].recallAtK, 1.0);
  assert.equal(report.cases[1].faithfulness, 0.0);
  assert.equal(report.cases[1].contextPrecision, 0.5);
  assert.equal(report.cases[1].recallAtK, 1.0);
});

test("createStudioServer serves live HTTP /health, /v1/evaluate, and SSE /v1/stream", async () => {
  const studio = createStudioServer();
  const { url } = await studio.listen(0, "127.0.0.1");

  try {
    const healthRes = await fetch(`${url}/health`);
    assert.equal(healthRes.status, 200);
    const health = await healthRes.json();
    assert.equal(health.status, "ok");

    const evalRes = await fetch(`${url}/v1/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cases: [
          {
            id: "http-1",
            query: "Castagnoli CRC32C ring header alignment",
            relevantDocIds: ["c1"],
            retrievedDocs: [
              { id: "c1", text: "Castagnoli CRC32C seals 64 byte cache line aligned ring headers." }
            ],
            generatedAnswer: "Castagnoli CRC32C seals 64 byte cache line aligned ring headers."
          }
        ]
      })
    });
    assert.equal(evalRes.status, 200);
    const evalJson = await evalRes.json();
    assert.equal(evalJson.ok, true);
    assert.equal(evalJson.report.faithfulness, 1.0);

    const streamRes = await fetch(`${url}/v1/stream?query=Castagnoli+CRC32C+ring+vault`);
    assert.equal(streamRes.status, 200);
    assert.match(streamRes.headers.get("content-type") ?? "", /text\/event-stream/);
    const sseText = await streamRes.text();
    assert.ok(sseText.includes("event: route"));
    assert.ok(sseText.includes("event: token"));
    assert.ok(sseText.includes("event: metrics"));
    assert.ok(sseText.includes("event: done"));
  } finally {
    await studio.close();
  }
});

test("RegressionDiffAnalyzer detects pairwise A/B chunking & prompt wins vs regressions", async () => {
  const baseline = [
    {
      id: "ab-1",
      query: "How does RRF combine BM25 and cosine rankings?",
      relevantDocIds: ["d1", "d2"],
      retrievedDocs: [
        { id: "noise-1", text: "Unrelated container image layer caching policy." },
        { id: "d1", text: "Reciprocal rank fusion merges lexical BM25 rankings with dense cosine similarity." }
      ],
      generatedAnswer: "Reciprocal rank fusion merges lexical BM25 rankings with dense cosine similarity."
    },
    {
      id: "ab-2",
      query: "How does P2 estimate tail quantiles?",
      relevantDocIds: ["p2-doc"],
      retrievedDocs: [
        { id: "p2-doc", text: "Jain Chlamtac P2 estimators track p95 and p99 quantiles in constant O(1) memory." }
      ],
      generatedAnswer: "Jain Chlamtac P2 estimators track p95 and p99 quantiles in constant O(1) memory."
    }
  ];

  const improvedCandidate = [
    {
      id: "ab-1",
      query: "How does RRF combine BM25 and cosine rankings?",
      relevantDocIds: ["d1", "d2"],
      retrievedDocs: [
        { id: "d1", text: "Reciprocal rank fusion merges lexical BM25 rankings with dense cosine similarity using k equals 60." },
        { id: "d2", text: "SQLite WAL FTS5 serves concurrent readers alongside dense vector scoring." }
      ],
      generatedAnswer:
        "Reciprocal rank fusion merges lexical BM25 rankings with dense cosine similarity. SQLite WAL FTS5 serves concurrent readers alongside dense vector scoring."
    },
    baseline[1]
  ];

  const analyzer = new RegressionDiffAnalyzer({ topK: 2 });
  const winReport = await analyzer.compare(baseline, improvedCandidate);
  assert.equal(winReport.passed, true);
  assert.equal(winReport.verdict, "PASS");
  assert.equal(winReport.caseBreakdown.improved, 1);
  assert.equal(winReport.caseBreakdown.regressed, 0);
  assert.ok(winReport.deltas.contextPrecision > 0.2);
  assert.ok(winReport.deltas.recallAtK > 0.2);
  assert.ok(winReport.drift.meanChunkOverlapJaccard !== null && winReport.drift.meanChunkOverlapJaccard < 1.0);
  assert.ok(formatDiffReport(winReport, "markdown").includes("PASS"));

  // Regressed candidate drops grounded claim -> fails faithfulness & case regression gate
  const regressedCandidate = [
    baseline[0],
    {
      ...baseline[1],
      retrievedDocs: [{ id: "wrong-doc", text: "CSS grid layout responsive media queries." }],
      generatedAnswer: "Superconducting dilution refrigerators cool transmon qubits below twenty millikelvin."
    }
  ];
  const failReport = await analyzer.compare(baseline, regressedCandidate);
  assert.equal(failReport.passed, false);
  assert.equal(failReport.verdict, "REGRESSION_DETECTED");
  assert.equal(failReport.caseBreakdown.regressed, 1);
  assert.ok(failReport.regressions[0].reasons.includes("faithfulness_drop"));
  assert.ok(failReport.regressions[0].reasons.includes("recall_chunk_miss"));
  assert.ok(failReport.gateFailures.length >= 2);
});

test("POST /v1/diff validates schema, scrubs PII/keys, and returns pairwise A/B diff JSON", async () => {
  const studio = createStudioServer({ topK: 2 });
  const { url } = await studio.listen(0, "127.0.0.1");
  const dynKey = ["sk", "abdiff9876543210token"].join("-");

  try {
    const baseCases = [
      {
        id: "http-diff-1",
        query: `Check CRC32C ring alignment with ${dynKey}`,
        relevantDocIds: ["r1"],
        retrievedDocs: [{ id: "r1", text: "Castagnoli CRC32C protects 64 byte cache line ring headers." }],
        generatedAnswer: "Castagnoli CRC32C protects 64 byte cache line ring headers."
      }
    ];
    const res = await fetch(`${url}/v1/diff`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ baseline: baseCases, candidate: baseCases })
    });
    assert.equal(res.status, 200);
    const payload = await res.json();
    assert.equal(payload.ok, true);
    assert.equal(payload.redactionCount, 2);
    assert.equal(payload.diff.passed, true);
    assert.equal(payload.diff.matchedCases, 1);
  } finally {
    await studio.close();
  }
});

test("CLI bin/cli.mjs executes diff & eval subcommands with proper CI exit codes (0 vs 1)", async () => {
  const tmp = await mkdtemp(join(tmpdir(), "rag-studio-cli-"));
  const basePath = join(tmp, "baseline.json");
  const passPath = join(tmp, "candidate_pass.json");
  const failPath = join(tmp, "candidate_fail.json");
  const cliPath = new URL("../bin/cli.mjs", import.meta.url).pathname;

  const baseData = [
    {
      id: "cli-1",
      query: "Sparse Top-2 MoE routing under VRAM cap",
      relevantDocIds: ["m1"],
      retrievedDocs: [
        { id: "noise", text: "Irrelevant CSS typography token guide." },
        { id: "m1", text: "Sparse top two MoE gating filters experts by VRAM cap and latency SLA." }
      ],
      generatedAnswer: "Sparse top two MoE gating filters experts by VRAM cap and latency SLA."
    }
  ];
  const passData = [
    {
      ...baseData[0],
      retrievedDocs: [
        { id: "m1", text: "Sparse top two MoE gating filters experts by VRAM cap and latency SLA." }
      ]
    }
  ];
  const failData = [
    {
      ...baseData[0],
      retrievedDocs: [{ id: "noise", text: "Irrelevant CSS typography token guide." }],
      generatedAnswer: "Unrelated hallucinated statement with zero lexical overlap."
    }
  ];

  try {
    await writeFile(basePath, JSON.stringify(baseData, null, 2), "utf8");
    await writeFile(passPath, JSON.stringify(passData, null, 2), "utf8");
    await writeFile(failPath, JSON.stringify(failData, null, 2), "utf8");

    const { stdout: jsonOut } = await execFileAsync(process.execPath, [
      cliPath,
      "diff",
      basePath,
      passPath,
      "--format",
      "json"
    ]);
    const parsed = JSON.parse(jsonOut);
    assert.equal(parsed.passed, true);
    assert.equal(parsed.caseBreakdown.improved, 1);

    const { stdout: evalOut } = await execFileAsync(process.execPath, [cliPath, "eval", passPath]);
    assert.match(evalOut, /Faithfulness=1\.0000/);

    let exitCode = 0;
    try {
      await execFileAsync(process.execPath, [cliPath, "diff", basePath, failPath]);
    } catch (err) {
      exitCode = err.code;
    }
    assert.equal(exitCode, 1);
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});

test("SlidingWindowTokenRateLimiter enforces exact request & token budgets without penalizing rejected calls", () => {
  let clock = 1_000;
  const limiter = new SlidingWindowTokenRateLimiter({
    windowMs: 60_000,
    maxRequestsPerWindow: 3,
    maxTokensPerWindow: 1_000,
    maxTrackedClients: 2,
    nowFn: () => clock
  });

  // t=1000: client-a consumes 600 tokens (remaining: 400 tokens, 2 requests)
  const d1 = limiter.consume("client-a", 600);
  assert.equal(d1.allowed, true);
  assert.equal(d1.remainingTokens, 400);
  assert.equal(d1.remainingRequests, 2);

  // t=5000: client-a consumes 300 tokens (used: 900, remaining: 100 tokens, 1 request)
  clock = 5_000;
  const d2 = limiter.consume("client-a", 300);
  assert.equal(d2.allowed, true);
  assert.equal(d2.remainingTokens, 100);

  // t=8000: client-a asks for 500 tokens -> deficit 400 tokens; oldest 600-token entry expires at 61,000 -> retryAfterMs = 53,000
  clock = 8_000;
  const blockedTok = limiter.consume("client-a", 500);
  assert.equal(blockedTok.allowed, false);
  assert.equal(blockedTok.reason, "sliding_window_tokens_exceeded");
  assert.equal(blockedTok.retryAfterMs, 53_000);
  const hdrs = limiter.toHeaders(blockedTok);
  assert.equal(hdrs["Retry-After"], "53");

  // Rejected call did NOT consume remaining 100 tokens -> a 100-token call still succeeds!
  const d3 = limiter.consume("client-a", 100);
  assert.equal(d3.allowed, true);
  assert.equal(d3.remainingTokens, 0);
  assert.equal(d3.remainingRequests, 0);

  // Next 1-token call hits request cap; oldest entry expires at 61,000 -> retryAfterMs = 53,000
  const blockedReq = limiter.consume("client-a", 1);
  assert.equal(blockedReq.allowed, false);
  assert.equal(blockedReq.retryAfterMs, 53_000);

  // Independent client-b has an isolated window; client-c evicts LRU client-a when cap=2
  assert.equal(limiter.consume("client-b", 250).allowed, true);
  assert.equal(limiter.consume("client-c", 250).allowed, true);
  assert.equal(limiter.snapshot().activeClients, 2);

  // Reconcile adjusts latest entry token count cleanly
  limiter.reconcile("client-c", -50);
  assert.equal(limiter.check("client-c", 1).usedTokens, 200);

  // Advance to 61,000 -> client-b (recorded at t=8,000) still has 250 tokens; at 68,000 -> slides to 0
  clock = 61_000;
  assert.equal(limiter.check("client-b", 250).usedTokens, 250);
  clock = 68_000;
  assert.equal(limiter.check("client-b", 250).usedTokens, 0);
});

test("OtlpTraceExporter propagates W3C traceparent, scrubs secrets, and exports OTLP/JSON spans & histograms", () => {
  let ns = 1_700_000_000_000_000_000n;
  const hooked = [];
  const exporter = new OtlpTraceExporter({
    serviceName: "rag-studio-test",
    serviceVersion: "1.2.0",
    maxSpans: 4,
    nowNsFn: () => ns,
    onSpan: (s) => hooked.push(s)
  });

  const upstreamTp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
  const parsed = exporter.parseTraceparent(upstreamTp);
  assert.ok(parsed);
  assert.equal(parsed.traceId, "4bf92f3577b34da6a3ce929d0e0e4736");
  assert.equal(exporter.parseTraceparent("00-00000000000000000000000000000000-00f067aa0ba902b7-01"), null);

  const dynKey = ["sk", "otlpsecret9876543210key"].join("-");
  const span = exporter.startSpan("HTTP POST /v1/query", {
    kind: 2,
    traceparent: upstreamTp,
    metricRoute: "/v1/query",
    attributes: { "rag.query.preview": `key ${dynKey}`, "rag.top_k": 3 }
  });
  assert.equal(span.traceId, "4bf92f3577b34da6a3ce929d0e0e4736");
  assert.equal(span.parentSpanId, "00f067aa0ba902b7");

  ns += 2_400_000n; // +2.4 ms
  span.addEvent("cache.lookup", { hit: true });
  const finished = span.end({ "http.status_code": 200 });
  assert.equal(hooked.length, 1);
  assert.equal(finished.endTimeUnixNano, String(ns));

  const otlp = exporter.exportOtlpJson();
  const serialized = JSON.stringify(otlp);
  assert.ok(!serialized.includes(dynKey));
  assert.ok(serialized.includes("[REDACTED_API_KEY]"));
  assert.equal(otlp.resourceSpans[0].scopeSpans[0].spans.length, 1);
  const histDp = otlp.resourceMetrics[0].scopeMetrics[0].metrics[0].histogram.dataPoints[0];
  assert.equal(histDp.count, "1");
  assert.equal(histDp.sum, 2.4);
});

test("Gateway /v1/query, /v1/stream, and /v1/traces enforce sliding-window rate limits and W3C OTLP tracing", async () => {
  let clock = 20_000;
  const studio = createStudioServer({
    rateLimiterOptions: {
      windowMs: 60_000,
      maxRequestsPerWindow: 10,
      maxTokensPerWindow: 300,
      nowFn: () => clock
    }
  });
  const { url } = await studio.listen(0, "127.0.0.1");
  const traceId = "abcdef0123456789abcdef0123456789";
  const parentSpanId = "1122334455667788";
  const traceparent = `00-${traceId}-${parentSpanId}-01`;

  try {
    // 1. POST /v1/query consumes 140 tokens -> 200 OK + grounded citations + W3C traceId propagation
    const qRes = await fetch(`${url}/v1/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Client-Id": "tenant-alpha",
        traceparent
      },
      body: JSON.stringify({
        query: "How does Castagnoli CRC32C protect 64 byte ring slot headers?",
        estimatedTokens: 140,
        topK: 2
      })
    });
    assert.equal(qRes.status, 200);
    assert.equal(qRes.headers.get("x-ratelimit-remaining-tokens"), "160");
    assert.equal(qRes.headers.get("x-trace-id"), traceId);
    const qJson = await qRes.json();
    assert.equal(qJson.ok, true);
    assert.equal(qJson.traceId, traceId);
    assert.equal(qJson.citations[0].id, "doc-crc32c-ring");

    // 2. GET /v1/stream consumes 160 tokens -> reaches exact 300/300 token budget -> 200 SSE
    clock = 25_000;
    const sOk = await fetch(
      `${url}/v1/stream?query=Jain+Chlamtac+P2+quantiles&estimatedTokens=160`,
      { headers: { "X-Client-Id": "tenant-alpha" } }
    );
    assert.equal(sOk.status, 200);
    assert.equal(sOk.headers.get("x-ratelimit-remaining-tokens"), "0");
    const sseBody = await sOk.text();
    assert.ok(sseBody.includes("event: done"));

    // 3. Next request on /v1/query or /v1/stream exceeds 300-token sliding window -> 429 + Retry-After
    clock = 30_000;
    const qBlocked = await fetch(`${url}/v1/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Client-Id": "tenant-alpha" },
      body: JSON.stringify({ query: "Blocked burst query", estimatedTokens: 100 })
    });
    assert.equal(qBlocked.status, 429);
    assert.equal(qBlocked.headers.get("retry-after"), "50"); // (20,000 + 60,000 - 30,000) / 1000 = 50s
    const blockedJson = await qBlocked.json();
    assert.equal(blockedJson.error, "rate_limit_exceeded");
    assert.equal(blockedJson.rateLimit.retryAfterMs, 50_000);

    const sBlocked = await fetch(`${url}/v1/stream?query=Blocked+SSE&estimatedTokens=50`, {
      headers: { "X-Client-Id": "tenant-alpha" }
    });
    assert.equal(sBlocked.status, 429);

    // 4. Different tenant is unaffected -> 200 OK
    const otherRes = await fetch(`${url}/v1/query?query=Reciprocal+rank+fusion+BM25&estimatedTokens=120`, {
      headers: { "X-Client-Id": "tenant-beta" }
    });
    assert.equal(otherRes.status, 200);

    // 5. GET /v1/traces exports OTLP/JSON spans & rate-limit metrics
    const trRes = await fetch(`${url}/v1/traces`);
    assert.equal(trRes.status, 200);
    const otlp = await trRes.json();
    assert.equal(otlp.summary.totalSpansRecorded, 5);
    assert.equal(otlp.summary.errorSpans, 2);
    assert.equal(otlp.summary.rateLimitAllowed, 3);
    assert.equal(otlp.summary.rateLimitRejected, 2);
  } finally {
    await studio.close();
  }
});

test("CLI v1.2.0 --otlp-out exports valid OpenTelemetry OTLP/JSON trace artifact", async () => {
  const tmp = await mkdtemp(join(tmpdir(), "rag-studio-otlp-cli-"));
  const suitePath = join(tmp, "suite.json");
  const otlpPath = join(tmp, "trace.otlp.json");
  const cliPath = new URL("../bin/cli.mjs", import.meta.url).pathname;

  try {
    await writeFile(
      suitePath,
      JSON.stringify([
        {
          id: "otlp-cli-1",
          query: "OpenTelemetry OTLP JSON span export",
          relevantDocIds: ["d1"],
          retrievedDocs: [{ id: "d1", text: "OpenTelemetry OTLP JSON span exporters propagate W3C traceparent headers." }],
          generatedAnswer: "OpenTelemetry OTLP JSON span exporters propagate W3C traceparent headers."
        }
      ]),
      "utf8"
    );

    const { stdout: verOut } = await execFileAsync(process.execPath, [cliPath, "--version"]);
    assert.equal(verOut.trim(), "1.2.0");

    await execFileAsync(process.execPath, [cliPath, "eval", suitePath, "--otlp-out", otlpPath]);
    const otlpJson = JSON.parse(await readFile(otlpPath, "utf8"));
    assert.equal(otlpJson.summary.serviceVersion, "1.2.0");
    assert.equal(otlpJson.resourceSpans[0].scopeSpans[0].spans[0].name, "cli.eval");
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});
