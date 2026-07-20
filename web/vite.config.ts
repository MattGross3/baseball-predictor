import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // The FastAPI backend runs on :8000 - proxy /api/* in dev so the
      // frontend never needs CORS or a hardcoded absolute URL. Same
      // pattern nginx.conf uses in production (see Dockerfile).
      '/api': {
        // 127.0.0.1, not localhost: Node's DNS resolution can prefer the
        // IPv6 loopback (::1) for "localhost" on Windows, which the
        // FastAPI backend (bound to 127.0.0.1 specifically) never
        // answers on - that mismatch shows up as a silent 502 from the
        // proxy, not a clear connection-refused error.
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
