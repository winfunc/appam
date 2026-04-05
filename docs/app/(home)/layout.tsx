import { HomeLayout } from "fumadocs-ui/layouts/home";
import Image from "next/image";
import type { ReactNode } from "react";

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <HomeLayout
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
        url: "/",
      }}
      links={[
        { text: "Docs", url: "/docs" },
        {
          text: "Quickstart",
          url: "/docs/getting-started/quickstart",
        },
        {
          text: "Examples",
          url: "/docs/examples/coding-agent-openai",
        },
      ]}
      githubUrl="https://github.com/winfunc/appam"
    >
      {children}
    </HomeLayout>
  );
}
