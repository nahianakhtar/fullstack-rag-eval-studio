import type { Server } from "node:http";

export interface JsonSchema {
  type?: "object" | "array" | "string" | "number" | "integer" | "boolean";
  required?: string[];
  additionalProperties?: boolean;
  properties?: Record<string, JsonSchema>;
  items?: JsonSchema;
  enum?: unknown[];
  minLength?: number;
  maxLength?: number;
  minimum?: number;
  maximum?: number;
  pattern?: string;
}

export interface ValidationResult<T = unknown> {
  valid: boolean;
  errors: string[];
  sanitized: T;
  redactionCount: number;
}

export declare class SchemaToolValidator {
  redact<T>(value: T): { sanitized: T; redactionCount: number };
  validate(value: unknown, schema: JsonSchema, path?: string): string[];
  validateAndRedact<T>(value: T, schema: JsonSchema): ValidationResult<T>;
}

export interface SemanticCacheOptions {
  maxEntries?: number;
  ttlMs?: number;
  similarityThreshold?: number;
  dimensions?: number;
  nowFn?: () => number;
}

export interface SemanticCacheLookup<V> {
  hit: boolean;
  matchType: "exact" | "semantic" | "miss";
  similarity: number;
  value?: V;
}

export interface SemanticCacheStats {
  size: number;
  maxEntries: number;
  exactHits: number;
  semanticHits: number;
  misses: number;
  evictions: number;
  hitRate: number;
}

export declare class SemanticLruCache<V = unknown> {
  constructor(options?: SemanticCacheOptions);
  set(query: string, value: V): void;
  get(query: string): SemanticCacheLookup<V>;
  stats(): SemanticCacheStats;
}

export interface CircuitBreakerOptions {
  windowMs?: number;
  maxTokensPerWindow?: number;
  maxCostUsdPerWindow?: number;
  softFallbackRatio?: number;
  failureThreshold?: number;
  cooldownMs?: number;
  primaryModel?: string;
  fallbackModel?: string;
  nowFn?: () => number;
}

export interface RouteDecision {
  allowed: boolean;
  selectedModel: string;
  tier: "primary" | "fallback" | "blocked";
  state: "CLOSED" | "OPEN" | "HALF_OPEN";
  windowTokens: number;
  windowCostUsd: number;
  reason: string;
}

export declare class TokenBudgetCircuitBreaker {
  constructor(options?: CircuitBreakerOptions);
  route(request?: { estimatedTokens?: number; estimatedCostUsd?: number }): RouteDecision;
  record(outcome: { model?: string; tokens?: number; costUsd?: number; ok: boolean }): void;
  snapshot(): {
    state: "CLOSED" | "OPEN" | "HALF_OPEN";
    consecutiveFailures: number;
    windowTokens: number;
    windowCostUsd: number;
    primaryModel: string;
    fallbackModel: string;
  };
}

export interface SlidingWindowRateLimiterOptions {
  windowMs?: number;
  maxRequestsPerWindow?: number;
  maxTokensPerWindow?: number;
  maxTrackedClients?: number;
  nowFn?: () => number;
}

export interface RateLimitDecision {
  allowed: boolean;
  clientId: string;
  limitRequests: number;
  remainingRequests: number;
  limitTokens: number;
  remainingTokens: number;
  usedRequests: number;
  usedTokens: number;
  requestedTokens: number;
  retryAfterMs: number;
  resetAfterMs: number;
  reason:
    | "allowed"
    | "sliding_window_tokens_exceeded"
    | "sliding_window_requests_exceeded"
    | "request_exceeds_window_capacity";
}

export interface RateLimiterSnapshot {
  activeClients: number;
  windowMs: number;
  maxRequestsPerWindow: number;
  maxTokensPerWindow: number;
  totalAllowed: number;
  totalRejected: number;
  totalTokensConsumed: number;
}

export declare class SlidingWindowTokenRateLimiter {
  constructor(options?: SlidingWindowRateLimiterOptions);
  check(clientId?: string, estimatedTokens?: number): RateLimitDecision;
  consume(clientId?: string, tokens?: number): RateLimitDecision;
  reconcile(clientId?: string, deltaTokens?: number): void;
  toHeaders(decision: RateLimitDecision): Record<string, string>;
  snapshot(): RateLimiterSnapshot;
}

export interface TraceContext {
  version: "00";
  traceId: string;
  spanId: string;
  traceFlags: string;
}

export interface OtlpAnyValue {
  stringValue?: string;
  boolValue?: boolean;
  intValue?: string;
  doubleValue?: number;
  arrayValue?: { values: OtlpAnyValue[] };
}

export interface OtlpAttribute {
  key: string;
  value: OtlpAnyValue;
}

export interface OtlpSpanEvent {
  timeUnixNano: string;
  name: string;
  attributes: OtlpAttribute[];
}

export interface OtlpSpan {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  kind: number;
  startTimeUnixNano: string;
  endTimeUnixNano: string;
  attributes: OtlpAttribute[];
  events: OtlpSpanEvent[];
  status: { code: number; message?: string };
}

export interface OtlpExporterStats {
  serviceName: string;
  serviceVersion: string;
  spansBuffered: number;
  maxSpans: number;
  totalSpansRecorded: number;
  droppedSpans: number;
  errorSpans: number;
  rateLimitAllowed: number;
  rateLimitRejected: number;
}

export interface OtlpExportPayload {
  resourceSpans: Array<{
    resource: { attributes: OtlpAttribute[] };
    scopeSpans: Array<{
      scope: { name: string; version: string };
      spans: OtlpSpan[];
    }>;
  }>;
  resourceMetrics: Array<{
    resource: { attributes: OtlpAttribute[] };
    scopeMetrics: Array<{
      scope: { name: string; version: string };
      metrics: unknown[];
    }>;
  }>;
  summary: OtlpExporterStats;
}

export interface OtlpTraceExporterOptions {
  serviceName?: string;
  serviceVersion?: string;
  scopeName?: string;
  maxSpans?: number;
  onSpan?: (span: OtlpSpan) => void;
  nowNsFn?: () => bigint | number;
  validator?: SchemaToolValidator;
}

export interface StartSpanOptions {
  kind?: number;
  parentContext?: TraceContext | null;
  traceparent?: string;
  metricRoute?: string;
  attributes?: Record<string, unknown>;
}

export interface ActiveSpan {
  readonly traceId: string;
  readonly spanId: string;
  readonly parentSpanId: string;
  readonly traceFlags: string;
  readonly traceparent: string;
  setAttribute(key: string, value: unknown): void;
  setAttributes(attrs: Record<string, unknown>): void;
  addEvent(name: string, attributes?: Record<string, unknown>): void;
  setStatus(code: 0 | 1 | 2, message?: string): void;
  end(endAttributes?: Record<string, unknown>): OtlpSpan;
}

export declare class OtlpTraceExporter {
  static readonly HISTOGRAM_BOUNDS_MS: readonly number[];
  constructor(options?: OtlpTraceExporterOptions);
  parseTraceparent(header?: string | null): TraceContext | null;
  formatTraceparent(ctx: { traceId: string; spanId: string; traceFlags?: string }): string;
  recordRateLimitDecision(allowed: boolean): void;
  startSpan(name: string, options?: StartSpanOptions): ActiveSpan;
  exportOtlpJson(): OtlpExportPayload;
  stats(): OtlpExporterStats;
  clear(): void;
}

export interface RetrievedChunk {
  id: string;
  text: string;
}

export interface GoldenEvalCase {
  id: string;
  query: string;
  relevantDocIds: string[];
  retrievedDocs: RetrievedChunk[];
  generatedAnswer: string;
}

export interface CaseEvalMetrics {
  id: string;
  faithfulness: number;
  contextPrecision: number;
  recallAtK: number;
  latencyMs: number;
}

export interface GoldenEvalReport {
  totalCases: number;
  topK: number;
  faithfulness: number;
  contextPrecision: number;
  recallAtK: number;
  meanLatencyMs: number;
  p50LatencyMs: number;
  p95LatencyMs: number;
  casesPerSec: number;
  cases: CaseEvalMetrics[];
}

export declare class GoldenEvalRunner {
  constructor(options?: { topK?: number });
  evaluateCase(item: GoldenEvalCase, k?: number): Omit<CaseEvalMetrics, "latencyMs">;
  runSuite(
    cases: GoldenEvalCase[],
    pipelineFn?: (item: GoldenEvalCase) => Promise<Partial<GoldenEvalCase>> | Partial<GoldenEvalCase>
  ): Promise<GoldenEvalReport>;
}

export interface RegressionDiffOptions {
  topK?: number;
  maxFaithfulnessDrop?: number;
  maxContextPrecisionDrop?: number;
  maxRecallDrop?: number;
  maxCaseRegressions?: number;
  caseEpsilon?: number;
  maxP95LatencyRatio?: number | null;
}

export type RegressionSuiteInput =
  | GoldenEvalCase[]
  | { topK?: number; cases: GoldenEvalCase[] }
  | GoldenEvalReport;

export interface CaseDiffEntry {
  id: string;
  status: "improved" | "regressed" | "unchanged";
  compositeDelta: number;
  faithfulnessDelta: number;
  contextPrecisionDelta: number;
  recallAtKDelta: number;
  reasons: string[];
  chunkOverlapJaccard: number | null;
  chunkWordsDelta: number | null;
  answerCosineSimilarity: number | null;
}

export interface SuiteDriftMetrics {
  meanChunkOverlapJaccard: number | null;
  meanChunkWordsDelta: number | null;
  meanAnswerCosineSimilarity: number | null;
  pairedStats: {
    meanDelta: number;
    stdDevDelta: number;
    cohensD: number;
    pairedTStat: number;
    winRate: number;
  };
}

export interface RegressionDiffReport {
  passed: boolean;
  verdict: "PASS" | "REGRESSION_DETECTED";
  gateFailures: string[];
  matchedCases: number;
  addedCaseIds: string[];
  removedCaseIds: string[];
  baseline: {
    totalCases: number;
    faithfulness: number;
    contextPrecision: number;
    recallAtK: number;
    p50LatencyMs: number;
    p95LatencyMs: number;
  };
  candidate: {
    totalCases: number;
    faithfulness: number;
    contextPrecision: number;
    recallAtK: number;
    p50LatencyMs: number;
    p95LatencyMs: number;
  };
  deltas: {
    faithfulness: number;
    contextPrecision: number;
    recallAtK: number;
    p50LatencyMs: number;
    p95LatencyMs: number;
  };
  drift: SuiteDriftMetrics;
  caseBreakdown: {
    improved: number;
    regressed: number;
    unchanged: number;
  };
  regressions: CaseDiffEntry[];
  improvements: CaseDiffEntry[];
}

export declare class RegressionDiffAnalyzer {
  constructor(options?: RegressionDiffOptions);
  compare(baselineInput: RegressionSuiteInput, candidateInput: RegressionSuiteInput): Promise<RegressionDiffReport>;
}

export declare function formatDiffReport(
  diff: RegressionDiffReport,
  format?: "table" | "json" | "markdown"
): string;

export interface StudioServerHandle {
  server: Server;
  validator: SchemaToolValidator;
  cache: SemanticLruCache;
  breaker: TokenBudgetCircuitBreaker;
  rateLimiter: SlidingWindowTokenRateLimiter;
  telemetry: OtlpTraceExporter;
  evaluator: GoldenEvalRunner;
  differ: RegressionDiffAnalyzer;
  listen(port?: number, host?: string): Promise<{ port: number; host: string; url: string }>;
  close(): Promise<void>;
}

export declare function createStudioServer(options?: {
  cacheOptions?: SemanticCacheOptions;
  breakerOptions?: CircuitBreakerOptions;
  rateLimiterOptions?: SlidingWindowRateLimiterOptions;
  telemetryOptions?: OtlpTraceExporterOptions;
  diffOptions?: RegressionDiffOptions;
  corpus?: RetrievedChunk[];
  topK?: number;
}): StudioServerHandle;
