import { source } from "@/lib/source";
import type { Graph } from "@/components/graph-view";

/**
 * Builds a graph of all documentation pages. Links are derived from the
 * page tree hierarchy (folder → child pages) so the graph is meaningful
 * even without the `extractLinkReferences` postprocess step.
 */
export function buildGraph(): Graph {
  const pages = source.getPages();
  const graph: Graph = { links: [], nodes: [] };

  for (const page of pages) {
    graph.nodes.push({
      id: page.url,
      url: page.url,
      text: page.data.title,
      description: page.data.description,
    });
  }

  // Derive links from URL hierarchy — /docs/a/b links to /docs/a
  const urlSet = new Set(pages.map((p) => p.url));

  for (const page of pages) {
    // Link to extracted references when available
    const refs = (page.data as unknown as Record<string, unknown>).extractedReferences;
    if (Array.isArray(refs)) {
      for (const ref of refs) {
        if (ref && typeof ref === "object" && "href" in ref) {
          const target = pages.find((p) => p.url === (ref as { href: string }).href);
          if (target) {
            graph.links.push({ source: page.url, target: target.url });
          }
        }
      }
    }

    // Fall back to parent-child hierarchy links
    const segments = page.url.split("/").filter(Boolean);
    if (segments.length >= 2) {
      const parentUrl = "/" + segments.slice(0, -1).join("/");
      if (urlSet.has(parentUrl)) {
        graph.links.push({ source: parentUrl, target: page.url });
      }
    }
  }

  return graph;
}
