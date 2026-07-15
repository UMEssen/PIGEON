import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Backend target for the dev proxy – override with VITE_BACKEND_URL env var
const backendUrl = process.env.VITE_BACKEND_URL || 'http://localhost:8003'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '0.0.0.0',
    port: 3005,
    proxy: {
      '/api': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/ws': {
        target: backendUrl.replace(/^http/, 'ws'),
        ws: true,
      },
    },
  },
})
