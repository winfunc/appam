import { source } from "@/lib/source";
import {
  DocsPage,
  DocsBody,
  DocsDescription,
  DocsTitle,
} from "fumadocs-ui/page";
import defaultMdxComponents from "fumadocs-ui/mdx";
import { notFound } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import { WordTicker } from "@/app/components/word-ticker";

const heroCards = [
  {
    title: "Quickstart",
    description: "Build and run your first agent in under five minutes.",
    href: "/docs/getting-started/quickstart",
  },
  {
    title: "Core Concepts",
    description: "Agents, tools, providers, streaming, and sessions.",
    href: "/docs/core-concepts/agents",
  },
  {
    title: "Examples",
    description: "Full agent examples across all supported providers.",
    href: "/docs/examples/coding-agent-openai",
  },
];

const capabilities = [
  "8 Providers",
  "Typed Tools",
  "Streaming",
  "Sessions",
  "Tracing",
  "Retries",
];

function DocsHomeHero() {
  return (
    <section className="appam-hero not-prose">
      <div className="appam-hero-content">
        <p className="appam-hero-badge">
          Open-source from{" "}
          <a href="https://winfunc.com" className="appam-hero-badge-link">
            Winfunc Research
          </a>
        </p>

        <h1 className="appam-hero-heading">
          Build <WordTicker /> agents
          <br />
          in <span className="appam-hero-rust">Rust</span>.
        </h1>

        <p className="appam-hero-description">
          Multi-provider LLM support, typed tools, real-time streaming,
          and session persistence&thinsp;&mdash;&thinsp;in one coherent crate.
        </p>

        <div className="appam-hero-caps">
          {capabilities.map((cap) => (
            <span key={cap} className="appam-hero-cap">
              {cap}
            </span>
          ))}
        </div>

        <div className="appam-hero-actions">
          <Link
            href="/docs/getting-started/quickstart"
            className="appam-hero-cta appam-hero-cta-primary"
          >
            Get started
            <span className="appam-hero-cta-arrow" aria-hidden="true">
              →
            </span>
          </Link>
          <Link
            href="/docs/examples/coding-agent-openai"
            className="appam-hero-cta appam-hero-cta-secondary"
          >
            View examples
          </Link>
        </div>
      </div>

      <div className="appam-hero-visual">
        <Image
          src="/appam-logo.png"
          alt="Appam mascot"
          width={340}
          height={340}
          priority
        />
      </div>

      <div className="appam-hero-code">
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

      <div className="appam-hero-cards">
        {heroCards.map((item) => (
          <Link key={item.href} href={item.href} className="appam-hero-card">
            <h3>
              {item.title}
              <span className="appam-hero-card-arrow" aria-hidden="true">
                →
              </span>
            </h3>
            <p>{item.description}</p>
          </Link>
        ))}
      </div>
    </section>
  );
}

export default async function Page(props: {
  params: Promise<{ slug?: string[] }>;
}) {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  const isDocsHome = !params.slug || params.slug.length === 0;
  const MDX = page.data.body;
  const toc = isDocsHome ? undefined : page.data.toc;

  return (
    <DocsPage toc={toc}>
      {isDocsHome ? (
        <DocsHomeHero />
      ) : (
        <>
          <DocsTitle>{page.data.title}</DocsTitle>
          <DocsDescription>{page.data.description}</DocsDescription>
        </>
      )}
      <DocsBody>
        <div className={isDocsHome ? "appam-home-body" : undefined}>
          <MDX components={{ ...defaultMdxComponents }} />
        </div>
      </DocsBody>
    </DocsPage>
  );
}

export async function generateStaticParams() {
  return source.generateParams();
}

export function generateMetadata(props: {
  params: Promise<{ slug?: string[] }>;
}) {
  return props.params.then((params) => {
    const page = source.getPage(params.slug);
    if (!page) notFound();
    return {
      title: page.data.title,
      description: page.data.description,
    };
  });
}
