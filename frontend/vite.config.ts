import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

/** In development the API runs on :8000 (`python -m pharmavision serve`); `/api` reaches it. */
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", rewrite: (path) => path.replace(/^\/api/, "") },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
  },
});
