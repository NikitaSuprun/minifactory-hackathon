import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev-only Basic Auth header so HMR (port 5173) can reach the FastAPI API (8041),
// which is behind HTTP Basic Auth. Uses the throwaway hackathon credentials.
const devAuth = "Basic " + Buffer.from("admin:123123").toString("base64");
const apiPaths = [
  "/status",
  "/logs",
  "/connect",
  "/disconnect",
  "/teleop",
  "/inference",
  "/camera.mjpeg",
  "/wrist.mjpeg",
];

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: Object.fromEntries(
      apiPaths.map((p) => [
        p,
        {
          target: "http://localhost:8041",
          changeOrigin: true,
          headers: { Authorization: devAuth },
        },
      ]),
    ),
  },
});
