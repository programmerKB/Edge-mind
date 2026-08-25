/** @file Vite development server and same-origin API proxy configuration. */

import { existsSync } from 'node:fs'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  // Docker resolves the Compose service name; native development reaches the
  // backend through the host loopback address.
  const defaultProxyTarget = existsSync('/.dockerenv')
    ? 'http://backend:8000'
    : 'http://127.0.0.1:8000'

  return {
    plugins: [react()],
    server: {
      host: '0.0.0.0',
      port: 5173,
      proxy: {
        '/api': {
          target: env.VITE_PROXY_TARGET || defaultProxyTarget,
          changeOrigin: true,
        },
      },
    },
  }
})
