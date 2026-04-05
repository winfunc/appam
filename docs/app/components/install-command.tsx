"use client";

import { useState } from "react";

export function InstallCommand() {
  const [copied, setCopied] = useState(false);

  const copy = () => {
    navigator.clipboard.writeText("cargo add appam");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={copy}
      className="appam-land-install"
      title="Copy to clipboard"
      type="button"
    >
      <span className="appam-land-install-prompt" aria-hidden="true">
        $
      </span>
      <code>cargo add appam</code>
      <span className="appam-land-install-copy">{copied ? "Copied" : "Copy"}</span>
    </button>
  );
}
