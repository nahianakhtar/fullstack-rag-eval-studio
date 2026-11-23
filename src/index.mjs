import { createServer } from "node:http";
import { performance } from "node:perf_hooks";

const SECRET_PATTERNS = [
  { re: /Bearer\s+[A-Za-z0-9._\-+/=]{8,}/gi, token: "Bearer [REDACTED_TOKEN]" },
  { re: /\b(?:sk|rk|pk)-[A-Za-z0-9_\-]{12,}\b/g, token: "[REDACTED_API_KEY]" },
  { re: /\bgh[pousr]_[A-Za-z0-9]{20,}\b/g, token: "[REDACTED_GH_TOKEN]" },
  { re: /\bAKIA[0-9A-Z]{16}\b/g, token: "[REDACTED_AWS_KEY]" },
  { re: /\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b/g, token: "[REDACTED_EMAIL]" },
  { re: /\b\d{3}-\d{2}-\d{4}\b/g, token: "[REDACTED_SSN]" }
];

const STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
  "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this",
  "to", "using", "via", "was", "were", "with"
]);

/**
 * Strict recursive JSON-Schema validator paired with deterministic PII/key redaction.
 */
export class SchemaToolValidator {
  redact(value) {
    let redactionCount = 0;
    const scrub = (node) => {
      if (typeof node === "string") {
        let out = node;
        for (const { re, token } of SECRET_PATTERNS) {
          out = out.replace(re, () => {
            redactionCount += 1;
            return token;
          });
        }
        return out;
      }
      if (Array.isArray(node)) {
        return node.map((item) => scrub(item));
      }
      if (node !== null && typeof node === "object") {
        const copy = {};
        for (const [k, v] of Object.entries(node)) {
          copy[k] = scrub(v);
        }
        return copy;
      }
      return node;
    };
    return { sanitized: scrub(value), redactionCount };
  }

  validate(value, schema, path = "$") {
    const errors = [];
    if (!schema || typeof schema !== "object") {
      return errors;
    }

    if (schema.type) {
      const okType = this.#matchesType(value, schema.type);
      if (!okType) {
        errors.push(`${path}: expected type '${schema.type}'`);
        return errors;
      }
    }

    if (Array.isArray(schema.enum) && !schema.enum.includes(value)) {
      errors.push(`${path}: value not in allowed enum [${schema.enum.join(", ")}]`);
    }

    if (typeof value === "string") {
      if (typeof schema.minLength === "number" && value.length < schema.minLength) {
        errors.push(`${path}: string length ${value.length} < minLength ${schema.minLength}`);
      }
      if (typeof schema.maxLength === "number" && value.length > schema.maxLength) {
        errors.push(`${path}: string length ${value.length} > maxLength ${schema.maxLength}`);
      }
      if (typeof schema.pattern === "string" && !(new RegExp(schema.pattern).test(value))) {
        errors.push(`${path}: string does not match pattern '${schema.pattern}'`);
      }
    }

    if (typeof value === "number") {
      if (typeof schema.minimum === "number" && value < schema.minimum) {
        errors.push(`${path}: ${value} < minimum ${schema.minimum}`);
      }
      if (typeof schema.maximum === "number" && value > schema.maximum) {
        errors.push(`${path}: ${value} > maximum ${schema.maximum}`);
      }
    }

    if (Array.isArray(value) && schema.items) {
      value.forEach((elem, idx) => {
        errors.push(...this.validate(elem, schema.items, `${path}[${idx}]`));
      });
    }

    if (value !== null && typeof value === "object" && !Array.isArray(value)) {
      const props = schema.properties ?? {};
      if (Array.isArray(schema.required)) {
        for (const reqKey of schema.required) {
          if (!(reqKey in value) || value[reqKey] === undefined) {
            errors.push(`${path}.${reqKey}: missing required property`);
          }
        }
      }
      if (schema.additionalProperties === false) {
        for (const key of Object.keys(value)) {
          if (!(key in props)) {
            errors.push(`${path}.${key}: additional property not permitted`);
          }
        }
      }
      for (const [key, subSchema] of Object.entries(props)) {
        if (key in value && value[key] !== undefined) {
          errors.push(...this.validate(value[key], subSchema, `${path}.${key}`));
        }
      }
    }

    return errors;
  }

  validateAndRedact(value, schema) {
    const { sanitized, redactionCount } = this.redact(value);
    const errors = this.validate(sanitized, schema);
    return {
      valid: errors.length === 0,
      errors,
      sanitized,
      redactionCount
    };
  }

  #matchesType(val, expected) {
    switch (expected) {
      case "object":
        return val !== null && typeof val === "object" && !Array.isArray(val);
      case "array":
        return Array.isArray(val);
      case "string":
        return typeof val === "string";
      case "number":
        return typeof val === "number" && Number.isFinite(val);
      case "integer":
        return typeof val === "number" && Number.isInteger(val);
      case "boolean":
        return typeof val === "boolean";
      default:
        return true;
    }
  }
}

function normalizeText(text) {
  return String(text ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function embedNgramCosine(text, dims = 256) {
  const norm = normalizeText(text);
  const vec = new Float64Array(dims);
  if (!norm) {
    return vec;
  }
  const tokens = norm.split(" ");
  const features = [];
  for (const tok of tokens) {
    if (!STOPWORDS.has(tok)) {
      features.push(`w:${tok}`);
      if (tok.length > 4) {
        features.push(`s:${tok.slice(0, 4)}`);
      }
    }
  }
  const compact = norm.replace(/\s+/g, "_");
  for (let i = 0; i + 3 <= compact.length; i += 1) {
    features.push(`c:${compact.slice(i, i + 3)}`);
  }

  for (const feat of features) {
    let h = 2166136261;
    for (let i = 0; i < feat.length; i += 1) {
      h ^= feat.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    const idx = (h >>> 0) % dims;
    const sign = (h & 1) === 0 ? 1.0 : -1.0;
    const weight = feat.startsWith("w:") ? 2.0 : 0.65;
    vec[idx] += sign * weight;
  }

  let sumSq = 0.0;
  for (let i = 0; i < dims; i += 1) {
    sumSq += vec[i] * vec[i];
  }
  if (sumSq > 0) {
    const inv = 1.0 / Math.sqrt(sumSq);
    for (let i = 0; i < dims; i += 1) {
      vec[i] *= inv;
    }
  }
  return vec;
}

function dotCosine(a, b) {
  let dot = 0.0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
  }
  return dot;
}

/**
 * Cosine character/word n-gram semantic LRU cache with TTL expiry and hit-rate telemetry.
 */
export class SemanticLruCache {
  #entries = new Map();
  #maxEntries;
  #ttlMs;
  #similarityThreshold;
  #dimensions;
  #nowFn;
  #exactHits = 0;
  #semanticHits = 0;
  #misses = 0;
  #evictions = 0;

  constructor(options = {}) {
    this.#maxEntries = options.maxEntries ?? 256;
    this.#ttlMs = options.ttlMs ?? 60_000;
    this.#similarityThreshold = options.similarityThreshold ?? 0.84;
    this.#dimensions = options.dimensions ?? 256;
    this.#nowFn = options.nowFn ?? (() => Date.now());
  }

  set(query, value) {
    const now = this.#nowFn();
    this.#pruneExpired(now);
    const key = normalizeText(query);
    if (this.#entries.has(key)) {
      this.#entries.delete(key);
    } else if (this.#entries.size >= this.#maxEntries) {
      const oldestKey = this.#entries.keys().next().value;
      if (oldestKey !== undefined) {
        this.#entries.delete(oldestKey);
        this.#evictions += 1;
      }
    }
    this.#entries.set(key, {
      key,
      vector: embedNgramCosine(key, this.#dimensions),
      value,
      expiresAt: now + this.#ttlMs
    });
  }

  get(query) {
    const now = this.#nowFn();
    this.#pruneExpired(now);
    const key = normalizeText(query);
    const exact = this.#entries.get(key);
    if (exact) {
      this.#entries.delete(key);
      this.#entries.set(key, exact);
      this.#exactHits += 1;
      return { hit: true, matchType: "exact", similarity: 1.0, value: exact.value };
    }

    const qVec = embedNgramCosine(key, this.#dimensions);
    let bestEntry = null;
    let bestSim = -1.0;

    for (const entry of this.#entries.values()) {
      const sim = dotCosine(qVec, entry.vector);
      if (sim > bestSim) {
        bestSim = sim;
        bestEntry = entry;
      }
    }

    if (bestEntry && bestSim >= this.#similarityThreshold) {
      this.#entries.delete(bestEntry.key);
      this.#entries.set(bestEntry.key, bestEntry);
      this.#semanticHits += 1;
      return {
        hit: true,
        matchType: "semantic",
        similarity: Number(bestSim.toFixed(4)),
        value: bestEntry.value
      };
    }

    this.#misses += 1;
    return { hit: false, matchType: "miss", similarity: Math.max(0, Number(bestSim.toFixed(4))) };
  }

  stats() {
    this.#pruneExpired(this.#nowFn());
    const total = this.#exactHits + this.#semanticHits + this.#misses;
    return {
      size: this.#entries.size,
      maxEntries: this.#maxEntries,
      exactHits: this.#exactHits,
      semanticHits: this.#semanticHits,
      misses: this.#misses,
      evictions: this.#evictions,
      hitRate: total === 0 ? 0 : Number(((this.#exactHits + this.#semanticHits) / total).toFixed(4))
    };
  }

  #pruneExpired(now) {
    for (const [key, entry] of this.#entries.entries()) {
      if (entry.expiresAt <= now) {
        this.#entries.delete(key);
      }
    }
  }
}

/**
 * Sliding-window token & USD cost governor with automatic fallback model routing.
 */
export class TokenBudgetCircuitBreaker {
  #windowMs;
  #maxTokensPerWindow;
  #maxCostUsdPerWindow;
  #softFallbackRatio;
  #failureThreshold;
  #cooldownMs;
  #primaryModel;
  #fallbackModel;
  #nowFn;
  #ledger = [];
  #state = "CLOSED";
  #consecutiveFailures = 0;
  #openedAt = 0;

  constructor(options = {}) {
    this.#windowMs = options.windowMs ?? 60_000;
    this.#maxTokensPerWindow = options.maxTokensPerWindow ?? 50_000;
    this.#maxCostUsdPerWindow = options.maxCostUsdPerWindow ?? 2.50;
    this.#softFallbackRatio = options.softFallbackRatio ?? 0.75;
    this.#failureThreshold = options.failureThreshold ?? 3;
    this.#cooldownMs = options.cooldownMs ?? 15_000;
    this.#primaryModel = options.primaryModel ?? "frontier-dense-70b";
    this.#fallbackModel = options.fallbackModel ?? "local-sparse-moe-q4";
    this.#nowFn = options.nowFn ?? (() => Date.now());
  }

  route(request = {}) {
    const now = this.#nowFn();
    this.#pruneWindow(now);
    const estTokens = request.estimatedTokens ?? 256;
    const estCost = request.estimatedCostUsd ?? 0.002;

    if (this.#state === "OPEN" && now - this.#openedAt >= this.#cooldownMs) {
      this.#state = "HALF_OPEN";
    }

    const { windowTokens, windowCostUsd } = this.#totals();
    const projectedTokens = windowTokens + estTokens;
    const projectedCost = windowCostUsd + estCost;

    if (projectedTokens > this.#maxTokensPerWindow || projectedCost > this.#maxCostUsdPerWindow) {
      return {
        allowed: false,
        selectedModel: "none",
        tier: "blocked",
        state: this.#state,
        windowTokens,
        windowCostUsd: Number(windowCostUsd.toFixed(6)),
        reason: "sliding_window_budget_exceeded"
      };
    }

    if (this.#state === "OPEN") {
      return {
        allowed: true,
        selectedModel: this.#fallbackModel,
        tier: "fallback",
        state: this.#state,
        windowTokens,
        windowCostUsd: Number(windowCostUsd.toFixed(6)),
        reason: "primary_circuit_open_fallback"
      };
    }

    const softTokens = this.#maxTokensPerWindow * this.#softFallbackRatio;
    const softCost = this.#maxCostUsdPerWindow * this.#softFallbackRatio;
    if (projectedTokens >= softTokens || projectedCost >= softCost) {
      return {
        allowed: true,
        selectedModel: this.#fallbackModel,
        tier: "fallback",
        state: this.#state,
        windowTokens,
        windowCostUsd: Number(windowCostUsd.toFixed(6)),
        reason: "soft_budget_fallback_routed"
      };
    }

    return {
      allowed: true,
      selectedModel: this.#primaryModel,
      tier: "primary",
      state: this.#state,
      windowTokens,
      windowCostUsd: Number(windowCostUsd.toFixed(6)),
      reason: this.#state === "HALF_OPEN" ? "half_open_probe" : "nominal_primary"
    };
  }

  record({ tokens = 0, costUsd = 0, ok = true } = {}) {
    const now = this.#nowFn();
    this.#pruneWindow(now);
    this.#ledger.push({ ts: now, tokens, costUsd });

    if (ok) {
      this.#consecutiveFailures = 0;
      this.#state = "CLOSED";
    } else {
      this.#consecutiveFailures += 1;
      if (this.#state === "HALF_OPEN" || this.#consecutiveFailures >= this.#failureThreshold) {
        this.#state = "OPEN";
        this.#openedAt = now;
      }
    }
  }

  snapshot() {
    this.#pruneWindow(this.#nowFn());
    const { windowTokens, windowCostUsd } = this.#totals();
    return {
      state: this.#state,
      consecutiveFailures: this.#consecutiveFailures,
      windowTokens,
      windowCostUsd: Number(windowCostUsd.toFixed(6)),
      primaryModel: this.#primaryModel,
      fallbackModel: this.#fallbackModel
    };
  }

  #pruneWindow(now) {
    const cutoff = now - this.#windowMs;
    while (this.#ledger.length > 0 && this.#ledger[0].ts <= cutoff) {
      this.#ledger.shift();
    }
  }

  #totals() {
    let windowTokens = 0;
    let windowCostUsd = 0;
    for (const item of this.#ledger) {
      windowTokens += item.tokens;
      windowCostUsd += item.costUsd;
    }
    return { windowTokens, windowCostUsd };
  }
}

function contentWords(text) {
  return normalizeText(text)
    .split(" ")
    .filter((tok) => tok.length >= 2 && !STOPWORDS.has(tok));
}

function percentile(sortedVals, p) {
  if (sortedVals.length === 0) {
    return 0;
  }
  const idx = Math.min(sortedVals.length - 1, Math.max(0, Math.ceil(p * sortedVals.length) - 1));
  return sortedVals[idx];
}

/**
 * Deterministic RAG evaluation engine computing Faithfulness, Context Precision (AP@K),
 * Recall@K, and high-resolution p50/p95 latency metrics.
 */
export class GoldenEvalRunner {
  #topK;

  constructor(options = {}) {
    this.#topK = options.topK ?? 5;
  }

  evaluateCase(item, k = this.#topK) {
    const topDocs = (item.retrievedDocs ?? []).slice(0, k);
    const relevantSet = new Set(item.relevantDocIds ?? []);

    // 1. Recall@K
    let hits = 0;
    for (const doc of topDocs) {
      if (relevantSet.has(doc.id)) {
        hits += 1;
      }
    }
    const recallAtK = relevantSet.size === 0 ? 1.0 : hits / relevantSet.size;

    // 2. Context Precision (Average Precision @ K)
    let relevantSeen = 0;
    let precisionAccum = 0.0;
    topDocs.forEach((doc, idx) => {
      if (relevantSet.has(doc.id)) {
        relevantSeen += 1;
        precisionAccum += relevantSeen / (idx + 1);
      }
    });
    const denom = Math.min(relevantSet.size, topDocs.length);
    const contextPrecision =
      relevantSeen === 0 || denom === 0 ? 0.0 : precisionAccum / relevantSeen;

    // 3. Claim-level Faithfulness against retrieved context tokens
    const contextTokenSet = new Set(
      contentWords(topDocs.map((d) => d.text).join(" "))
    );
    const sentences = String(item.generatedAnswer ?? "")
      .split(/[.!?]+/)
      .map((s) => s.trim())
      .filter(Boolean);

    let faithfulness = 0.0;
    if (sentences.length > 0) {
      let supportedSentences = 0;
      for (const sentence of sentences) {
        const words = contentWords(sentence);
        if (words.length === 0) {
          supportedSentences += 1;
          continue;
        }
        let grounded = 0;
        for (const w of words) {
          if (contextTokenSet.has(w)) {
            grounded += 1;
          }
        }
        if (grounded / words.length >= 0.6) {
          supportedSentences += 1;
        }
      }
      faithfulness = supportedSentences / sentences.length;
    }

    return {
      id: item.id,
      faithfulness: Number(faithfulness.toFixed(4)),
      contextPrecision: Number(contextPrecision.toFixed(4)),
      recallAtK: Number(recallAtK.toFixed(4))
    };
  }

  async runSuite(cases, pipelineFn) {
    const caseMetrics = [];
    const latencies = [];
    const suiteStart = performance.now();

    for (const rawCase of cases) {
      const t0 = performance.now();
      let enriched = rawCase;
      if (typeof pipelineFn === "function") {
        const patch = await pipelineFn(rawCase);
        enriched = { ...rawCase, ...patch };
      }
      const scored = this.evaluateCase(enriched, this.#topK);
      const latencyMs = Number((performance.now() - t0).toFixed(4));
      latencies.push(latencyMs);
      caseMetrics.push({ ...scored, latencyMs });
    }

    const elapsedSec = Math.max(1e-6, (performance.now() - suiteStart) / 1000.0);
    const n = Math.max(1, caseMetrics.length);
    const sortedLat = [...latencies].sort((a, b) => a - b);

    const mean = (key) =>
      Number((caseMetrics.reduce((acc, c) => acc + c[key], 0) / n).toFixed(4));

    return {
      totalCases: caseMetrics.length,
      topK: this.#topK,
      faithfulness: mean("faithfulness"),
      contextPrecision: mean("contextPrecision"),
      recallAtK: mean("recallAtK"),
      meanLatencyMs: mean("latencyMs"),
      p50LatencyMs: Number(percentile(sortedLat, 0.50).toFixed(4)),
      p95LatencyMs: Number(percentile(sortedLat, 0.95).toFixed(4)),
      casesPerSec: Number((caseMetrics.length / elapsedSec).toFixed(1)),
      cases: caseMetrics
    };
  }
}

const EVALUATE_REQUEST_SCHEMA = {
  type: "object",
  required: ["cases"],
  additionalProperties: false,
  properties: {
    topK: { type: "integer", minimum: 1, maximum: 50 },
    cases: {
      type: "array",
      items: {
        type: "object",
        required: ["id", "query", "relevantDocIds", "retrievedDocs", "generatedAnswer"],
        additionalProperties: false,
        properties: {
          id: { type: "string", minLength: 1 },
          query: { type: "string", minLength: 1 },
          relevantDocIds: { type: "array", items: { type: "string" } },
          retrievedDocs: {
            type: "array",
            items: {
              type: "object",
              required: ["id", "text"],
              additionalProperties: false,
              properties: {
                id: { type: "string", minLength: 1 },
                text: { type: "string" }
              }
            }
          },
          generatedAnswer: { type: "string" }
        }
      }
    }
  }
};

function readJsonBody(req, maxBytes = 1_048_576) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let total = 0;
    req.on("data", (chunk) => {
      total += chunk.length;
      if (total > maxBytes) {
        reject(new Error("payload_too_large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      try {
        const text = Buffer.concat(chunks).toString("utf8");
        resolve(text ? JSON.parse(text) : {});
      } catch {
        reject(new Error("invalid_json"));
      }
    });
    req.on("error", reject);
  });
}

function sendJson(res, statusCode, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
    "X-Content-Type-Options": "nosniff"
  });
  res.end(body);
}

/**
 * Creates a zero-dependency Node.js 22 HTTP + SSE streaming studio gateway.
 */
export function createStudioServer(options = {}) {
  const validator = new SchemaToolValidator();
  const cache = new SemanticLruCache(options.cacheOptions);
  const breaker = new TokenBudgetCircuitBreaker(options.breakerOptions);
  const evaluator = new GoldenEvalRunner({ topK: options.topK ?? 5 });

  const server = createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");

    if (req.method === "GET" && url.pathname === "/health") {
      return sendJson(res, 200, {
        status: "ok",
        service: "fullstack-rag-eval-studio",
        version: "1.0.0",
        cache: cache.stats(),
        breaker: breaker.snapshot()
      });
    }

    if (req.method === "POST" && url.pathname === "/v1/evaluate") {
      try {
        const body = await readJsonBody(req);
        const checked = validator.validateAndRedact(body, EVALUATE_REQUEST_SCHEMA);
        if (!checked.valid) {
          return sendJson(res, 400, { ok: false, errors: checked.errors });
        }
        const report = await evaluator.runSuite(checked.sanitized.cases);
        return sendJson(res, 200, {
          ok: true,
          redactionCount: checked.redactionCount,
          report
        });
      } catch (err) {
        return sendJson(res, 400, { ok: false, errors: [err.message] });
      }
    }

    if ((req.method === "GET" || req.method === "POST") && url.pathname === "/v1/stream") {
      let rawQuery = url.searchParams.get("query") ?? "hybrid rag bm25 cosine rrf architecture";
      if (req.method === "POST") {
        try {
          const body = await readJsonBody(req);
          if (typeof body.query === "string" && body.query.trim()) {
            rawQuery = body.query;
          }
        } catch (err) {
          return sendJson(res, 400, { ok: false, errors: [err.message] });
        }
      }

      const { sanitized: cleanQuery, redactionCount } = validator.redact(rawQuery);
      const routeInfo = breaker.route({ estimatedTokens: 160, estimatedCostUsd: 0.001 });
      if (!routeInfo.allowed) {
        return sendJson(res, 429, { ok: false, route: routeInfo });
      }

      res.writeHead(200, {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no"
      });

      const writeSse = (event, data) => {
        res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
      };

      const lookup = cache.get(cleanQuery);
      writeSse("route", {
        selectedModel: routeInfo.selectedModel,
        tier: routeInfo.tier,
        cacheHit: lookup.hit,
        matchType: lookup.matchType,
        redactionCount
      });

      const answerText = lookup.hit
        ? lookup.value
        : `Hybrid retrieval fused BM25 lexical recall with cosine vector similarity for "${cleanQuery}".`;
      if (!lookup.hit) {
        cache.set(cleanQuery, answerText);
      }

      const words = answerText.split(" ");
      for (let i = 0; i < words.length; i += 3) {
        writeSse("token", {
          index: i / 3,
          delta: words.slice(i, i + 3).join(" ") + (i + 3 < words.length ? " " : "")
        });
      }

      breaker.record({ model: routeInfo.selectedModel, tokens: 96, costUsd: 0.0008, ok: true });
      writeSse("metrics", { cache: cache.stats(), breaker: breaker.snapshot() });
      writeSse("done", { done: true });
      res.end();
      return;
    }

    return sendJson(res, 404, { ok: false, errors: ["route_not_found"] });
  });

  return {
    server,
    validator,
    cache,
    breaker,
    evaluator,
    listen(port = 0, host = "127.0.0.1") {
      return new Promise((resolve, reject) => {
        server.once("error", reject);
        server.listen(port, host, () => {
          const addr = server.address();
          const boundPort = typeof addr === "object" && addr ? addr.port : port;
          resolve({ port: boundPort, host, url: `http://${host}:${boundPort}` });
        });
      });
    },
    close() {
      return new Promise((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    }
  };
    }
