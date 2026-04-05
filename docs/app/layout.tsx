import "./global.css";
import type { Metadata } from "next";
import { DM_Serif_Display, Manrope, JetBrains_Mono } from "next/font/google";
import type { ReactNode } from "react";
import { RootProvider } from "fumadocs-ui/provider";

const dmSerifDisplay = DM_Serif_Display({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-appam-display",
  display: "swap",
});

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-appam-sans",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-appam-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "Appam",
    template: "%s — Appam",
  },
  description:
    "Rust framework for building AI agents with multi-provider LLM support, streaming, typed tools, and session persistence.",
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={`${dmSerifDisplay.variable} ${manrope.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
      <body className="appam-body">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
