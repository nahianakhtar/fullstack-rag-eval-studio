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

export interface StudioServerHandle {
  server: Server;
  validator: SchemaToolValidator;
  cache: SemanticLruCache;
  breaker: TokenBudgetCircuitBreaker;
  evaluator: GoldenEvalRunner;
  listen(port?: number, host?: string): Promise<{ port: number; host: string; url: string }>;
  close(): Promise<void>;
}

export declare function createStudioServer(options?: {
  cacheOptions?: SemanticCacheOptions;
  breakerOptions?: CircuitBreakerOptions;
  topK?: number;
}): StudioServerHandle;
