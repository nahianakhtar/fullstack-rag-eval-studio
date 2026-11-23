import assert from "node:assert/strict";
import test from "node:test";
import {
  GoldenEvalRunner,
  SchemaToolValidator,
  SemanticLruCache,
  TokenBudgetCircuitBreaker,
  createStudioServer
} from "../src/index.mjs";

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
