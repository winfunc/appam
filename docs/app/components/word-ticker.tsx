"use client";

import { useEffect, useState } from "react";

const WORDS = ["long-horizon", "reliable", "parallel"];
const INTERVAL_MS = 2600;
const EXIT_MS = 280;

export function WordTicker() {
  const [index, setIndex] = useState(0);
  const [phase, setPhase] = useState<"in" | "out">("in");

  useEffect(() => {
    const id = setInterval(() => {
      setPhase("out");
      setTimeout(() => {
        setIndex((prev) => (prev + 1) % WORDS.length);
        setPhase("in");
      }, EXIT_MS);
    }, INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  return (
    <span className="appam-ticker-wrap" aria-live="polite" aria-atomic="true">
      {/* All words rendered in the same grid cell — widest one sets the width */}
      {WORDS.map((word, i) => {
        const isActive = i === index;
        let cls = "appam-ticker-word";
        if (isActive) {
          cls += phase === "in" ? " appam-ticker-in" : " appam-ticker-out";
        } else {
          cls += " appam-ticker-hidden";
        }
        return (
          <span key={word} className={cls} aria-hidden={!isActive}>
            {word}
          </span>
        );
      })}
    </span>
  );
}
