import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), serveOrtRuntime()],
  server: {
    port: 5174,
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
  },
  preview: {
    port: 4174,
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
  },
});

function serveOrtRuntime() {
  function stripOrtImportQuery(req, _res, next) {
    const url = new URL(req.url || "/", "http://localhost");
    if (url.pathname.startsWith("/ort/") && url.pathname.endsWith(".mjs") && url.searchParams.has("import")) {
      req.url = url.pathname;
    }
    next();
  }

  return {
    name: "serve-ort-runtime",
    configureServer(server) {
      server.middlewares.use(stripOrtImportQuery);
    },
    configurePreviewServer(server) {
      server.middlewares.use(stripOrtImportQuery);
    },
  };
}
