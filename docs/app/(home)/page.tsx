import Link from "next/link";
import Image from "next/image";
import { WordTicker } from "@/app/components/word-ticker";
import { InstallCommand } from "@/app/components/install-command";
import { FadeIn } from "@/app/components/fade-in";
import { GraphView } from "@/components/graph-view";
import { buildGraph } from "@/lib/build-graph";

/* ═══════════════════════════════════════════════════
   SVG Illustrations — Premium, schematic, technical
   ═══════════════════════════════════════════════════ */

function IllustProviders() {
  return (
    <svg viewBox="0 0 300 200" fill="none" aria-hidden="true" className="appam-land-svg">
      <defs>
        <radialGradient id="prov-glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="var(--appam-accent)" stopOpacity="0.15" />
          <stop offset="100%" stopColor="var(--appam-accent)" stopOpacity="0" />
        </radialGradient>
      </defs>
      {/* Background radial glow */}
      <circle cx="150" cy="100" r="95" fill="url(#prov-glow)" />

      {/* Connecting mesh — spread wide */}
      <path d="M150 100 L40 30 M150 100 L260 30 M150 100 L20 100 M150 100 L280 100 M150 100 L60 180 M150 100 L240 180 M150 100 L150 15 M150 100 L40 160 M150 100 L260 160"
        stroke="var(--appam-accent)" strokeWidth="1" strokeOpacity="0.18" strokeDasharray="4 4" />

      {/* Orbital paths */}
      <circle cx="150" cy="100" r="55" stroke="var(--appam-accent)" strokeWidth="1" strokeOpacity="0.08" />
      <circle cx="150" cy="100" r="90" stroke="var(--appam-accent)" strokeWidth="1" strokeOpacity="0.05" />

      {/* Center API hub */}
      <circle cx="150" cy="100" r="18" fill="var(--appam-surface-raised)" stroke="var(--appam-accent)" strokeWidth="2" />
      <circle cx="150" cy="100" r="7" fill="var(--appam-accent)" />

      {/* Satellite provider nodes — spread to edges */}
      <g stroke="var(--appam-accent)" strokeWidth="1.5" fill="var(--appam-surface-raised)">
        <circle cx="150" cy="15" r="10" />
        <circle cx="150" cy="15" r="4" fill="var(--appam-accent)" stroke="none" opacity="0.6" />

        <circle cx="40" cy="30" r="9" />
        <circle cx="40" cy="30" r="3.5" fill="var(--appam-accent)" stroke="none" opacity="0.5" />

        <circle cx="260" cy="30" r="11" />
        <circle cx="260" cy="30" r="4" fill="var(--appam-accent)" stroke="none" opacity="0.4" />

        <circle cx="20" cy="100" r="12" />
        <path d="M16 100 H24 M20 96 V104" stroke="var(--appam-accent)" strokeWidth="1" />

        <circle cx="280" cy="100" r="9" />
        <circle cx="280" cy="100" r="3.5" fill="var(--appam-accent)" stroke="none" opacity="0.5" />

        <circle cx="40" cy="160" r="10" />
        <circle cx="40" cy="160" r="4" fill="var(--appam-accent)" stroke="none" opacity="0.3" />

        <circle cx="260" cy="160" r="12" />
        <circle cx="260" cy="160" r="4" fill="var(--appam-accent)" stroke="none" opacity="0.7" />

        <circle cx="60" cy="180" r="9" />
        <circle cx="60" cy="180" r="3" fill="var(--appam-accent)" stroke="none" opacity="0.4" />

        <circle cx="240" cy="180" r="10" />
        <circle cx="240" cy="180" r="4" fill="var(--appam-accent)" stroke="none" opacity="0.6" />
      </g>
    </svg>
  );
}

function IllustTools() {
  return (
    <svg viewBox="0 0 200 120" fill="none" aria-hidden="true" className="appam-land-svg">
      {/* Abstract Editor / Struct definition */}
      <rect x="20" y="15" width="160" height="90" rx="6" stroke="var(--appam-accent)" strokeWidth="1.5" strokeOpacity="0.2" fill="var(--appam-surface-raised)" />
      {/* Top bar */}
      <path d="M20 35 L180 35" stroke="var(--appam-accent)" strokeWidth="1" strokeOpacity="0.1" />
      <circle cx="32" cy="25" r="2.5" fill="var(--appam-accent)" fillOpacity="0.2" />
      <circle cx="42" cy="25" r="2.5" fill="var(--appam-accent)" fillOpacity="0.2" />
      <circle cx="52" cy="25" r="2.5" fill="var(--appam-accent)" fillOpacity="0.2" />

      {/* Macro attribute #[tool] */}
      <rect x="30" y="45" width="35" height="10" rx="3" fill="var(--appam-accent)" fillOpacity="0.1" />

      {/* Struct definition line */}
      <rect x="30" y="60" width="20" height="6" rx="2" fill="var(--appam-accent)" fillOpacity="0.4" />
      <rect x="55" y="60" width="45" height="6" rx="2" fill="var(--appam-accent)" fillOpacity="0.8" />

      {/* Fields */}
      <rect x="40" y="75" width="30" height="4" rx="1.5" fill="var(--appam-accent)" fillOpacity="0.3" />
      <rect x="75" y="75" width="25" height="4" rx="1.5" fill="var(--appam-accent)" fillOpacity="0.15" />

      <rect x="40" y="85" width="40" height="4" rx="1.5" fill="var(--appam-accent)" fillOpacity="0.3" />
      <rect x="85" y="85" width="20" height="4" rx="1.5" fill="var(--appam-accent)" fillOpacity="0.15" />

      {/* Connection arrow pointing to a gear */}
      <path d="M125 70 L145 70" stroke="var(--appam-accent)" strokeWidth="1.5" strokeOpacity="0.3" strokeDasharray="3 3" />
      <circle cx="160" cy="70" r="10" stroke="var(--appam-accent)" strokeWidth="1.5" strokeOpacity="0.5" />
      <circle cx="160" cy="70" r="3" fill="var(--appam-accent)" fillOpacity="0.8" />
    </svg>
  );
}

function IllustStreaming() {
  return (
    <svg viewBox="0 0 200 120" fill="none" aria-hidden="true" className="appam-land-svg">
      {/* Grid background */}
      <g stroke="var(--appam-accent)" strokeOpacity="0.05" strokeWidth="1">
        <line x1="0" y1="30" x2="200" y2="30" />
        <line x1="0" y1="60" x2="200" y2="60" />
        <line x1="0" y1="90" x2="200" y2="90" />
        <line x1="50" y1="0" x2="50" y2="120" />
        <line x1="100" y1="0" x2="100" y2="120" />
        <line x1="150" y1="0" x2="150" y2="120" />
      </g>

      {/* Waveform 1 (Smooth) */}
      <path d="M -10 60 C 30 60, 40 20, 80 20 C 120 20, 140 80, 180 80 C 200 80, 210 60, 220 60"
        stroke="var(--appam-accent)" strokeWidth="2" strokeOpacity="0.3" fill="none" />

      {/* Waveform 2 (Sharp) */}
      <path d="M -10 80 L 20 80 L 40 40 L 70 90 L 100 50 L 130 90 L 150 40 L 180 60 L 210 60"
        stroke="var(--appam-accent)" strokeWidth="1.5" strokeOpacity="0.6" fill="none" strokeLinejoin="round" />

      {/* Playhead indicator */}
      <line x1="130" y1="10" x2="130" y2="110" stroke="var(--appam-accent)" strokeWidth="1" opacity="0.5" strokeDasharray="4 4" />
      <circle cx="130" cy="90" r="4" fill="var(--appam-surface-raised)" stroke="var(--appam-accent)" strokeWidth="1.5" />
      <circle cx="130" cy="20" r="4" fill="var(--appam-accent)" />
    </svg>
  );
}

function IllustSessions() {
  return (
    <svg viewBox="0 0 200 120" fill="none" aria-hidden="true" className="appam-land-svg">
      {/* Stacked isometric planes representing persisted states */}
      <g transform="translate(100, 45)">
        {/* Layer 3 (Bottom) */}
        <g transform="translate(0, 30)">
          <path d="M0 25 L-60 10 L0 -5 L60 10 Z" fill="var(--appam-accent)" fillOpacity="0.05" stroke="var(--appam-accent)" strokeOpacity="0.2" strokeWidth="1" />
          <path d="M0 25 L-60 10 L-60 16 L0 31 L60 16 L60 10 Z" fill="var(--appam-accent)" fillOpacity="0.1" />
        </g>

        {/* Layer 2 (Middle) */}
        <g transform="translate(0, 15)">
          <path d="M0 25 L-60 10 L0 -5 L60 10 Z" fill="var(--appam-surface-raised)" />
          <path d="M0 25 L-60 10 L0 -5 L60 10 Z" fill="var(--appam-accent)" fillOpacity="0.15" stroke="var(--appam-accent)" strokeOpacity="0.3" strokeWidth="1" />
          <path d="M0 25 L-60 10 L-60 16 L0 31 L60 16 L60 10 Z" fill="var(--appam-accent)" fillOpacity="0.2" />
        </g>

        {/* Layer 1 (Top / Active) */}
        <g transform="translate(0, 0)">
          <path d="M0 25 L-60 10 L0 -5 L60 10 Z" fill="var(--appam-surface-raised)" />
          <path d="M0 25 L-60 10 L0 -5 L60 10 Z" fill="var(--appam-accent)" fillOpacity="0.25" stroke="var(--appam-accent)" strokeOpacity="0.6" strokeWidth="1.5" />
          <path d="M0 25 L-60 10 L-60 16 L0 31 L60 16 L60 10 Z" fill="var(--appam-accent)" fillOpacity="0.35" />

          {/* Active indicator nodes on top surface */}
          <circle cx="0" cy="10" r="3" fill="var(--appam-surface-raised)" stroke="var(--appam-accent)" strokeWidth="1.5" />
          <circle cx="-25" cy="5" r="2" fill="var(--appam-accent)" opacity="0.6" />
          <circle cx="25" cy="15" r="2" fill="var(--appam-accent)" opacity="0.6" />
        </g>
      </g>

      {/* Resume curve pointing up */}
      <path d="M 165 95 C 185 95, 185 55, 165 55" stroke="var(--appam-accent)" strokeWidth="1.5" strokeOpacity="0.4" fill="none" strokeDasharray="4 4" />
      <polygon points="165,51 160,56 165,61" fill="var(--appam-accent)" opacity="0.6" />
    </svg>
  );
}

function IllustTracing() {
  return (
    <svg viewBox="0 0 200 120" fill="none" aria-hidden="true" className="appam-land-svg">
      {/* Gantt / Flame graph representation of a trace */}

      {/* Time axis */}
      <line x1="20" y1="15" x2="180" y2="15" stroke="var(--appam-accent)" strokeWidth="1" strokeOpacity="0.1" />
      <circle cx="20" cy="15" r="2" fill="var(--appam-accent)" opacity="0.2" />
      <circle cx="60" cy="15" r="2" fill="var(--appam-accent)" opacity="0.2" />
      <circle cx="100" cy="15" r="2" fill="var(--appam-accent)" opacity="0.2" />
      <circle cx="140" cy="15" r="2" fill="var(--appam-accent)" opacity="0.2" />
      <circle cx="180" cy="15" r="2" fill="var(--appam-accent)" opacity="0.2" />

      {/* Root span */}
      <rect x="20" y="30" width="160" height="12" rx="2" fill="var(--appam-accent)" fillOpacity="0.1" stroke="var(--appam-accent)" strokeOpacity="0.3" strokeWidth="1" />

      {/* Child span 1 */}
      <rect x="40" y="50" width="60" height="12" rx="2" fill="var(--appam-accent)" fillOpacity="0.3" />
      <path d="M30 42 L30 56 L38 56" stroke="var(--appam-accent)" strokeWidth="1" strokeOpacity="0.3" fill="none" />

      {/* Nested child */}
      <rect x="50" y="70" width="30" height="12" rx="2" fill="var(--appam-accent)" fillOpacity="0.6" />
      <path d="M45 62 L45 76 L48 76" stroke="var(--appam-accent)" strokeWidth="1" strokeOpacity="0.3" fill="none" />

      {/* Child span 2 */}
      <rect x="110" y="50" width="70" height="12" rx="2" fill="var(--appam-accent)" fillOpacity="0.2" />
      <path d="M30 56 L30 85" stroke="var(--appam-accent)" strokeWidth="1" strokeOpacity="0.1" fill="none" strokeDasharray="2 2" />
      <path d="M105 42 L105 56 L108 56" stroke="var(--appam-accent)" strokeWidth="1" strokeOpacity="0.3" fill="none" />

      {/* Child span 3 (Error/Highlight) */}
      <rect x="130" y="70" width="40" height="12" rx="2" fill="var(--appam-accent)" fillOpacity="0.8" />
      <path d="M115 62 L115 76 L128 76" stroke="var(--appam-accent)" strokeWidth="1" strokeOpacity="0.3" fill="none" />
      <circle cx="160" cy="76" r="3" fill="var(--appam-surface-raised)" />

      {/* Sub-child */}
      <rect x="140" y="90" width="20" height="12" rx="2" fill="var(--appam-accent)" fillOpacity="0.2" />
      <path d="M135 82 L135 96 L138 96" stroke="var(--appam-accent)" strokeWidth="1" strokeOpacity="0.3" fill="none" />
    </svg>
  );
}

function IllustReliability() {
  return (
    <svg viewBox="0 0 200 120" fill="none" aria-hidden="true" className="appam-land-svg">
      {/* Circular retry/resilience loop spanning horizontally */}

      {/* Left node */}
      <circle cx="50" cy="60" r="25" stroke="var(--appam-accent)" strokeOpacity="0.15" strokeWidth="2" strokeDasharray="6 4" />
      <circle cx="50" cy="60" r="15" fill="var(--appam-accent)" fillOpacity="0.05" />
      <circle cx="50" cy="60" r="4" fill="var(--appam-accent)" fillOpacity="0.4" />

      {/* Right node */}
      <circle cx="150" cy="60" r="25" stroke="var(--appam-accent)" strokeOpacity="0.15" strokeWidth="2" strokeDasharray="6 4" />
      <circle cx="150" cy="60" r="15" fill="var(--appam-accent)" fillOpacity="0.1" />
      <path d="M145 60 L148 64 L156 55" stroke="var(--appam-accent)" strokeWidth="2" fill="none" strokeOpacity="0.7" strokeLinecap="round" strokeLinejoin="round" />

      {/* Connecting loop */}
      <path d="M50 35 C80 35, 120 35, 150 35" stroke="var(--appam-accent)" strokeWidth="2" strokeOpacity="0.6" fill="none" />
      <polygon points="100,32 105,35 100,38" fill="var(--appam-accent)" opacity="0.8" />

      <path d="M150 85 C120 85, 80 85, 50 85" stroke="var(--appam-accent)" strokeWidth="2" strokeOpacity="0.3" fill="none" />
      <polygon points="100,82 95,85 100,88" fill="var(--appam-accent)" opacity="0.5" />

      {/* Exponential backoff representation below */}
      <path d="M 60 105 L 70 105" stroke="var(--appam-accent)" strokeWidth="2" strokeOpacity="0.2" strokeLinecap="round" />
      <path d="M 80 105 L 100 105" stroke="var(--appam-accent)" strokeWidth="2" strokeOpacity="0.4" strokeLinecap="round" />
      <path d="M 110 105 L 140 105" stroke="var(--appam-accent)" strokeWidth="2" strokeOpacity="0.7" strokeLinecap="round" />
    </svg>
  );
}

/* ═══════════════════════════════════════════════════
   Static data
   ═══════════════════════════════════════════════════ */

const features = [
  {
    accent: "provider",
    title: "Eight Providers, One API",
    description:
      "Anthropic, OpenAI, Vertex, OpenRouter, Azure, Bedrock, Codex — swap with a single line change. No vendor lock-in, no abstraction tax.",
    Illustration: IllustProviders,
  },
  {
    accent: "tools",
    title: "Typed Tool System",
    description:
      "Define tools as Rust structs with #[tool], closures, or TOML declarations. Full type safety at compile time, zero boilerplate at runtime.",
    Illustration: IllustTools,
  },
  {
    accent: "stream",
    title: "Streaming by Default",
    description:
      "Real-time events to console, channels, callbacks, or custom consumers. Token-by-token control over every response as it arrives.",
    Illustration: IllustStreaming,
  },
  {
    accent: "session",
    title: "Session Persistence",
    description:
      "Conversations survive restarts via SQLite. Resume, query, and inspect any session long after the agent has finished running.",
    Illustration: IllustSessions,
  },
  {
    accent: "trace",
    title: "Built-in Tracing",
    description:
      "JSONL traces, structured stream events, and SQLite-backed history give you full observability without bolting on external tooling.",
    Illustration: IllustTracing,
  },
  {
    accent: "reliable",
    title: "Production Reliability",
    description:
      "Retries with exponential backoff, continuation mechanics, rate limiting, and provider-specific tuning. Built for jobs that cannot fail silently.",
    Illustration: IllustReliability,
  },
];

const providers = [
  { name: "Anthropic", detail: "Messages API" },
  { name: "OpenAI", detail: "Responses API" },
  { name: "OpenRouter", detail: "Any model" },
  { name: "Google Vertex", detail: "Gemini" },
  { name: "Azure OpenAI", detail: "Responses" },
  { name: "Azure Anthropic", detail: "Messages" },
  { name: "AWS Bedrock", detail: "Messages" },
  { name: "OpenAI Codex", detail: "Responses" },
];

function HeroCodePreview() {
  return (
    <div className="appam-hero-code appam-land-code">
      <div className="appam-hero-code-chrome">
        <div className="appam-hero-code-dots">
          <span />
          <span />
          <span />
        </div>
        <span className="appam-hero-code-filename">main.rs</span>
      </div>
      <pre className="appam-hero-code-body">
        <code>
          <span className="appam-code-line">
            <span className="appam-code-keyword">use</span>{" "}
            <span className="appam-code-entity">appam</span>::prelude::*;
          </span>
          <span className="appam-code-line" />
          <span className="appam-code-line">
            <span className="appam-code-meta">#[tokio::main]</span>
          </span>
          <span className="appam-code-line">
            <span className="appam-code-keyword">async fn</span>{" "}
            <span className="appam-code-function">main</span>() -&gt;{" "}
            <span className="appam-code-entity">Result</span>&lt;()&gt;{" {"}
          </span>
          <span className="appam-code-line">
            {"    "}
            <span className="appam-code-keyword">let</span> agent ={" "}
            <span className="appam-code-entity">Agent</span>::
            <span className="appam-code-function">quick</span>(
          </span>
          <span className="appam-code-line">
            {"        "}
            <span className="appam-code-string">
              &quot;anthropic/claude-sonnet-4-5&quot;
            </span>
            ,
          </span>
          <span className="appam-code-line">
            {"        "}
            <span className="appam-code-string">
              &quot;You are a helpful assistant.&quot;
            </span>
            ,
          </span>
          <span className="appam-code-line">{"        "}vec![],</span>
          <span className="appam-code-line">{"    "})?;</span>
          <span className="appam-code-line" />
          <span className="appam-code-line">{"    "}agent</span>
          <span className="appam-code-line">
            {"        "}.
            <span className="appam-code-function">stream</span>(
            <span className="appam-code-string">
              &quot;Plan a release checklist&quot;
            </span>
            )
          </span>
          <span className="appam-code-line">
            {"        "}.
            <span className="appam-code-function">on_content</span>(|text|{" "}
            <span className="appam-code-function">print!</span>(
            <span className="appam-code-string">
              &quot;{"{"}{"}"}&quot;
            </span>
            , text))
          </span>
          <span className="appam-code-line">
            {"        "}.
            <span className="appam-code-function">run</span>()
          </span>
          <span className="appam-code-line">{"        "}.await?;</span>
          <span className="appam-code-line" />
          <span className="appam-code-line">
            {"    "}
            <span className="appam-code-entity">Ok</span>(())
          </span>
          <span className="appam-code-line">{"}"}</span>
        </code>
      </pre>
    </div>
  );
}

export default function LandingPage() {
  const graph = buildGraph();

  return (
    <main className="appam-landing">
      {/* ═══ Hero ═══ */}
      <section className="appam-land-hero">
        <div className="appam-land-hero-grid">
          <div className="appam-land-hero-text">
            <p className="appam-hero-badge" style={{ animationDelay: "0ms" }}>
              Open-source from{" "}
              <a href="https://winfunc.com" className="appam-hero-badge-link">
                Winfunc Research
              </a>
            </p>

            <h1 className="appam-land-h1">
              Build <WordTicker /> agents
              <br />
              in <span className="appam-hero-rust">Rust</span>.
            </h1>

            <p className="appam-land-desc">
              Multi-provider LLM support, typed tools, real-time streaming,
              and session persistence&thinsp;&mdash;&thinsp;in one coherent
              crate.
            </p>

            <div className="appam-land-install-row">
              <InstallCommand />
            </div>

            <div className="appam-hero-actions" style={{ animationDelay: "200ms" }}>
              <Link
                href="/docs/getting-started/quickstart"
                className="appam-hero-cta appam-hero-cta-primary"
              >
                Get started
                <span className="appam-hero-cta-arrow" aria-hidden="true">→</span>
              </Link>
              <Link
                href="https://github.com/winfunc/appam"
                className="appam-hero-cta appam-hero-cta-secondary"
              >
                View on GitHub
              </Link>
            </div>
          </div>

          <div className="appam-land-hero-visual">
            <div className="appam-land-hero-logo-wrap">
              <Image
                src="/appam-logo.png"
                alt="Appam"
                width={380}
                height={380}
                priority
                className="appam-land-hero-logo"
              />
            </div>
          </div>
        </div>

        <FadeIn delay={300}>
          <HeroCodePreview />
        </FadeIn>
      </section>

      {/* ═══ Features — Asymmetric Bento ═══ */}
      <section className="appam-land-section">
        <FadeIn>
          <p className="appam-land-label">Capabilities</p>
          <h2 className="appam-land-heading">
            Everything agents need to run in production.
          </h2>
        </FadeIn>

        <div className="appam-land-bento">
          {features.map((f, i) => (
            <FadeIn
              key={f.accent}
              delay={i * 70}
              className={`appam-bento-card appam-bento--${f.accent}`}
            >
              <div className="appam-bento-illust">
                <f.Illustration />
              </div>
              <div className="appam-bento-body">
                <h3 className="appam-bento-title">{f.title}</h3>
                <p className="appam-bento-desc">{f.description}</p>
              </div>
            </FadeIn>
          ))}
        </div>
      </section>

      {/* ═══ Providers ═══ */}
      <section className="appam-land-section">
        <FadeIn>
          <p className="appam-land-label">Provider Support</p>
          <h2 className="appam-land-heading">
            One crate. Eight providers. Zero friction.
          </h2>
          <p className="appam-land-subdesc">
            Switch providers with a one-line model string change. The streaming
            API, tool system, and session management remain identical across all
            of them.
          </p>
        </FadeIn>

        <div className="appam-land-providers">
          {providers.map((p, i) => (
            <FadeIn key={p.name} delay={i * 60} className="appam-land-provider">
              <span className="appam-land-provider-name">{p.name}</span>
              <span className="appam-land-provider-detail">{p.detail}</span>
            </FadeIn>
          ))}
        </div>
      </section>

      {/* ═══ Graph View ═══ */}
      <section className="appam-land-section">
        <FadeIn>
          <p className="appam-land-label">Documentation</p>
          <h2 className="appam-land-heading">Navigate the knowledge graph.</h2>
          <p className="appam-land-subdesc">
            Explore the full documentation as an interactive graph. Click any
            node to jump straight to that page.
          </p>
        </FadeIn>

        <FadeIn delay={100} className="appam-land-graph-wrap">
          <GraphView graph={graph} />
        </FadeIn>
      </section>

      {/* ═══ CTA ═══ */}
      <section className="appam-land-cta-section">
        <FadeIn>
          <h2 className="appam-land-cta-heading">Ready to build?</h2>
          <p className="appam-land-cta-desc">
            Go from zero to a working agent in under five minutes.
          </p>
          <div className="appam-land-cta-actions">
            <Link
              href="/docs/getting-started/quickstart"
              className="appam-hero-cta appam-hero-cta-primary"
            >
              Get started
              <span className="appam-hero-cta-arrow" aria-hidden="true">→</span>
            </Link>
            <Link href="/docs" className="appam-hero-cta appam-hero-cta-secondary">
              Read the docs
            </Link>
          </div>
        </FadeIn>
      </section>

      {/* ═══ Footer ═══ */}
      <footer className="appam-land-footer">
        <div className="appam-land-footer-inner">
          <div className="appam-land-footer-brand">
            <Image
              src="/appam-logo.png"
              alt="Appam"
              width={24}
              height={24}
              className="appam-land-footer-logo"
            />
            <span className="appam-land-footer-name">Appam</span>
          </div>
          <nav className="appam-land-footer-links">
            <Link href="/docs">Docs</Link>
            <Link href="/docs/getting-started/quickstart">Quickstart</Link>
            <Link href="/docs/api-reference/agent-builder">API Reference</Link>
            <a href="https://github.com/winfunc/appam">GitHub</a>
          </nav>
          <p className="appam-land-footer-copy">
            Built by{" "}
            <a href="https://winfunc.com">Winfunc Research</a>. Released
            under the MIT License.
          </p>
        </div>
      </footer>
    </main>
  );
}
