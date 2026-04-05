import { readFileSync } from "node:fs";
import path from "node:path";
import { DynamicCodeBlock } from "fumadocs-ui/components/dynamic-codeblock";

/**
 * Render the canonical Appam skill file as a single copyable Markdown block.
 *
 * The docs site uses `docs/SKILL.md` as the single source of truth for the
 * raw skill content so the Fumadocs page stays in sync with the standalone
 * file that agents can copy directly.
 */
export function SkillMarkdown() {
  const skillPath = path.join(process.cwd(), "SKILL.md");
  const code = readFileSync(skillPath, "utf8");

  return (
    <DynamicCodeBlock
      lang="md"
      code={code}
      codeblock={{
        title: "SKILL.md",
      }}
    />
  );
}
