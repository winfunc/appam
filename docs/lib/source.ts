import { docs } from "@/.source";
import { loader } from "fumadocs-core/source";

const raw = docs.toFumadocsSource();

export const source = loader({
  baseUrl: "/docs",
  source: {
    files: typeof raw.files === "function" ? (raw.files as () => typeof raw.files)() : raw.files,
  },
});
