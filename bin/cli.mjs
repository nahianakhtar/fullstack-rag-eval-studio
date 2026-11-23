#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  GoldenEvalRunner,
  OtlpTraceExporter,
  RegressionDiffAnalyzer,
  SchemaToolValidator,
  formatDiffReport
} from "../src/index.mjs";

const VERSION = "1.2.0";

const HELP_TEXT = `fullstack-rag-eval-studio v${VERSION}
Zero-dependency RAG evaluation, OTLP trace export & pairwise A/B prompt/chunking regression diff CLI.

Usage:
  fullstack-rag-eval-studio diff <baseline.json> <candidate.json> [options]
  fullstack-rag-eval-studio eval <suite.json> [options]

Options:
  --top-k <int>                  Top-K retrieved chunks to score (default: 5)
  --max-faithfulness-drop <num>  Allowed mean faithfulness drop (default: 0.01)
  --max-precision-drop <num>     Allowed mean AP@K drop (default: 0.02)
  --max-recall-drop <num>        Allowed mean Recall@K drop (default: 0.01)
  --max-case-regressions <int>   Allowed count of regressed cases (default: 0)
  --format <table|json|markdown> Output format (default: table)
  --otlp-out <path.json>         Export OpenTelemetry (OTLP/JSON) trace spans & histograms
  --no-fail                      Always exit 0 even when regressions are detected
  -v, --version                  Print CLI version
  -h, --help                     Show this help message
`;

function parseFlags(argv) {
  const positional = [];
  const flags = {
    topK: 5,
    maxFaithfulnessDrop: 0.01,
    maxContextPrecisionDrop: 0.02,
    maxRecallDrop: 0.01,
    maxCaseRegressions: 0,
    format: "table",
    otlpOut: null,
    failOnRegression: true,
    help: false,
    version: false
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "-h" || arg === "--help") {
      flags.help = true;
    } else if (arg === "-v" || arg === "--version") {
      flags.version = true;
    } else if (arg === "--no-fail") {
      flags.failOnRegression = false;
    } else if (arg === "--fail-on-regression") {
      flags.failOnRegression = true;
    } else if (arg === "--top-k") {
      flags.topK = Number.parseInt(argv[++i] ?? "", 10);
    } else if (arg === "--max-faithfulness-drop") {
      flags.maxFaithfulnessDrop = Number.parseFloat(argv[++i] ?? "");
    } else if (arg === "--max-precision-drop") {
      flags.maxContextPrecisionDrop = Number.parseFloat(argv[++i] ?? "");
    } else if (arg === "--max-recall-drop") {
      flags.maxRecallDrop = Number.parseFloat(argv[++i] ?? "");
    } else if (arg === "--max-case-regressions") {
      flags.maxCaseRegressions = Number.parseInt(argv[++i] ?? "", 10);
    } else if (arg === "--format") {
      flags.format = argv[++i] ?? "table";
    } else if (arg === "--otlp-out") {
      flags.otlpOut = argv[++i] ?? null;
      if (!flags.otlpOut) {
        throw new Error("missing_otlp_out_path");
      }
    } else if (arg.startsWith("-")) {
      throw new Error(`unknown_option:${arg}`);
    } else {
      positional.push(arg);
    }
  }

  if (!["table", "json", "markdown"].includes(flags.format)) {
    throw new Error(`invalid_format:${flags.format}`);
  }
  if (!Number.isInteger(flags.topK) || flags.topK < 1) {
    throw new Error("invalid_top_k");
  }
  return { positional, flags };
}

async function loadSanitizedJson(filePath, validator) {
  const rawText = await readFile(resolve(filePath), "utf8");
  const parsed = JSON.parse(rawText);
  return validator.redact(parsed).sanitized;
}

export async function runCli(argv = process.argv.slice(2), io = { stdout: console.log, stderr: console.error }) {
  let parsedArgs;
  try {
    parsedArgs = parseFlags(argv);
  } catch (err) {
    io.stderr(`Error: ${err.message}`);
    return 2;
  }

  const { positional, flags } = parsedArgs;
  if (flags.version) {
    io.stdout(VERSION);
    return 0;
  }
  if (flags.help || positional.length === 0) {
    io.stdout(HELP_TEXT.trimEnd());
    return positional.length === 0 && !flags.help ? 2 : 0;
  }

  const subcommand = positional[0];
  const validator = new SchemaToolValidator();
  const telemetry = new OtlpTraceExporter({ serviceVersion: VERSION, validator });

  if (subcommand === "diff") {
    if (positional.length < 3) {
      io.stderr("Error: diff requires <baseline.json> and <candidate.json>");
      return 2;
    }
    const span = telemetry.startSpan("cli.diff", {
      metricRoute: "cli.diff",
      attributes: { "cli.subcommand": "diff", "rag.top_k": flags.topK }
    });
    try {
      const [baselineData, candidateData] = await Promise.all([
        loadSanitizedJson(positional[1], validator),
        loadSanitizedJson(positional[2], validator)
      ]);
      const analyzer = new RegressionDiffAnalyzer({
        topK: flags.topK,
        maxFaithfulnessDrop: flags.maxFaithfulnessDrop,
        maxContextPrecisionDrop: flags.maxContextPrecisionDrop,
        maxRecallDrop: flags.maxRecallDrop,
        maxCaseRegressions: flags.maxCaseRegressions
      });
      const diff = await analyzer.compare(baselineData, candidateData);
      if (!diff.passed) {
        span.setStatus(2, "regression_detected");
      }
      span.end({
        "rag.diff.verdict": diff.verdict,
        "rag.diff.matched_cases": diff.matchedCases,
        "rag.diff.regressed_cases": diff.caseBreakdown.regressed
      });
      if (flags.otlpOut) {
        await writeFile(resolve(flags.otlpOut), JSON.stringify(telemetry.exportOtlpJson(), null, 2), "utf8");
      }
      io.stdout(formatDiffReport(diff, flags.format));
      return !diff.passed && flags.failOnRegression ? 1 : 0;
    } catch (err) {
      span.setStatus(2, err.message);
      span.end();
      io.stderr(`Error: ${err.message}`);
      return 2;
    }
  }

  if (subcommand === "eval") {
    if (positional.length < 2) {
      io.stderr("Error: eval requires <suite.json>");
      return 2;
    }
    const span = telemetry.startSpan("cli.eval", {
      metricRoute: "cli.eval",
      attributes: { "cli.subcommand": "eval", "rag.top_k": flags.topK }
    });
    try {
      const suiteData = await loadSanitizedJson(positional[1], validator);
      const cases = Array.isArray(suiteData) ? suiteData : suiteData?.cases;
      if (!Array.isArray(cases)) {
        span.setStatus(2, "invalid_suite_json");
        span.end();
        io.stderr("Error: suite JSON must be an array of cases or { cases: [...] }");
        return 2;
      }
      const runner = new GoldenEvalRunner({ topK: flags.topK });
      const report = await runner.runSuite(cases);
      span.end({
        "rag.eval.cases": report.totalCases,
        "rag.eval.faithfulness": report.faithfulness,
        "rag.eval.context_precision": report.contextPrecision,
        "rag.eval.recall_at_k": report.recallAtK
      });
      if (flags.otlpOut) {
        await writeFile(resolve(flags.otlpOut), JSON.stringify(telemetry.exportOtlpJson(), null, 2), "utf8");
      }
      if (flags.format === "json") {
        io.stdout(JSON.stringify(report, null, 2));
      } else {
        io.stdout(
          `Golden Suite (${report.totalCases} cases, topK=${report.topK}): Faithfulness=${report.faithfulness.toFixed(4)} AP@K=${report.contextPrecision.toFixed(4)} Recall@K=${report.recallAtK.toFixed(4)} p95=${report.p95LatencyMs.toFixed(4)}ms`
        );
      }
      return 0;
    } catch (err) {
      span.setStatus(2, err.message);
      span.end();
      io.stderr(`Error: ${err.message}`);
      return 2;
    }
  }

  io.stderr(`Error: unknown subcommand '${subcommand}'`);
  return 2;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const code = await runCli(process.argv.slice(2));
  process.exitCode = code;
}
