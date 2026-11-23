import { performance } from "node:perf_hooks";
import {
  GoldenEvalRunner,
  OtlpTraceExporter,
  SchemaToolValidator,
  SemanticLruCache,
  SlidingWindowTokenRateLimiter,
  TokenBudgetCircuitBreaker
} from "../src/index.mjs";

const TOPICS = [
  {
    id: "sys-ring",
    q: "How does the C11 FFI ring buffer prevent false sharing and silent bit rot?",
    doc: "Each ring slot header uses 64 byte cache line alignment and dual Castagnoli CRC32C checksums over header and payload.",
    ans: "Each ring slot header uses 64 byte cache line alignment and dual Castagnoli CRC32C checksums over header and payload."
  },
  {
    id: "rag-rrf",
    q: "How does hybrid search combine lexical BM25 and dense cosine similarity?",
    doc: "Reciprocal rank fusion merges SQLite FTS5 BM25 lexical rankings with L2 normalized cosine vector scores using constant k equals 60.",
    ans: "Reciprocal rank fusion merges SQLite FTS5 BM25 lexical rankings with L2 normalized cosine vector scores."
  },
  {
    id: "moe-router",
    q: "How are queries routed across local quantized models under VRAM constraints?",
    doc: "Sparse top two mixture of experts gating filters models by VRAM budget and latency SLA before computing softmax routing weights.",
    ans: "Sparse top two mixture of experts gating filters models by VRAM budget and latency SLA before computing softmax routing weights."
  },
  {
    id: "quantiles-p2",
    q: "How are p95 and p99 tail latencies estimated without unbounded sample arrays?",
    doc: "Jain Chlamtac piecewise parabolic P2 estimators maintain five markers in constant O(1) memory to track streaming quantiles.",
    ans: "Jain Chlamtac piecewise parabolic P2 estimators maintain five markers in constant O(1) memory to track streaming quantiles."
  }
];

const cases = [];
for (let i = 0; i < 120; i += 1) {
  const t = TOPICS[i % TOPICS.length];
  const distractor = TOPICS[(i + 1) % TOPICS.length];
  cases.push({
    id: `golden-${String(i + 1).padStart(3, "0")}`,
    query: `${t.q} [variant ${i % 6}]`,
    relevantDocIds: [`${t.id}-primary`],
    retrievedDocs:
      i % 20 === 19
        ? [
            { id: `${distractor.id}-noise`, text: distractor.doc },
            { id: `${t.id}-primary`, text: t.doc }
          ]
        : [
            { id: `${t.id}-primary`, text: t.doc },
            { id: `${distractor.id}-noise`, text: distractor.doc }
          ],
    generatedAnswer: t.ans
  });
}

const validator = new SchemaToolValidator();
const cache = new SemanticLruCache({ maxEntries: 64, similarityThreshold: 0.82 });
const breaker = new TokenBudgetCircuitBreaker({ maxTokensPerWindow: 500_000, maxCostUsdPerWindow: 25.0 });
const rateLimiter = new SlidingWindowTokenRateLimiter({
  windowMs: 60_000,
  maxRequestsPerWindow: 1_000,
  maxTokensPerWindow: 250_000
});
const telemetry = new OtlpTraceExporter({ serviceVersion: "1.2.0", maxSpans: 256, validator });
const runner = new GoldenEvalRunner({ topK: 2 });

const t0 = performance.now();
const report = await runner.runSuite(cases, (item) => {
  const span = telemetry.startSpan("bench.case", {
    metricRoute: "/v1/query",
    attributes: { "rag.case.id": item.id }
  });
  const checked = validator.redact(item.query);
  const rl = rateLimiter.consume("bench-runner", 120);
  telemetry.recordRateLimitDecision(rl.allowed);
  const cached = cache.get(checked.sanitized);
  if (!cached.hit) {
    cache.set(checked.sanitized, item.generatedAnswer);
  }
  const route = breaker.route({ estimatedTokens: 120, estimatedCostUsd: 0.0005 });
  breaker.record({ model: route.selectedModel, tokens: 120, costUsd: 0.0005, ok: true });
  span.end({ "rag.cache.hit": cached.hit, "rag.model.selected": route.selectedModel });
  return {};
});
const elapsedMs = performance.now() - t0;

const summary = {
  sdk: "fullstack-rag-eval-studio",
  version: "1.2.0",
  nodeRuntime: process.version,
  goldenSuite: {
    totalCases: report.totalCases,
    faithfulness: report.faithfulness,
    contextPrecision: report.contextPrecision,
    recallAtK: report.recallAtK,
    p50LatencyMs: report.p50LatencyMs,
    p95LatencyMs: report.p95LatencyMs,
    casesPerSec: report.casesPerSec,
    totalWallMs: Number(elapsedMs.toFixed(2))
  },
  semanticCache: cache.stats(),
  circuitBreaker: breaker.snapshot(),
  rateLimiter: rateLimiter.snapshot(),
  otlpTelemetry: telemetry.stats()
};

console.log(JSON.stringify(summary, null, 2));
