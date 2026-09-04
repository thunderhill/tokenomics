import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API base is baked in at build time (VITE_API_BASE_URL); docker-compose sets it
// to the host-reachable API URL since the browser, not the container network, makes
// these requests.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
