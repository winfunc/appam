import { source } from "@/lib/source";
import { DocsLayout } from "fumadocs-ui/layouts/docs";
import type { ReactNode } from "react";
import Image from "next/image";

const links = [
  {
    text: "Quickstart",
    url: "/docs/getting-started/quickstart",
    active: "nested-url" as const,
  },
  {
    text: "Guides",
    url: "/docs/guides/openai",
    active: "nested-url" as const,
  },
  {
    text: "API",
    url: "/docs/api-reference/agent-builder",
    active: "nested-url" as const,
  },
  {
    text: "Agent Skill",
    url: "/docs/guides/agent-skill",
    active: "nested-url" as const,
  },
];

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <DocsLayout
      tree={source.pageTree}
      githubUrl="https://github.com/winfunc/appam"
      links={links}
      nav={{
        title: (
          <div className="appam-wordmark">
            <Image
              src="/appam-logo.png"
              alt="Appam"
              width={32}
              height={32}
              className="appam-wordmark-logo"
            />
            <span className="appam-wordmark-name">Appam</span>
          </div>
        ),
        url: "/docs",
      }}
      sidebar={{ defaultOpenLevel: 1 }}
      themeSwitch={{ mode: "light-dark-system" }}
    >
      {children}
    </DocsLayout>
  );
}
