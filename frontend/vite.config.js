import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// defineConfig from 'vitest/config' (not plain 'vite') so the `test` block
// below type-checks and merges into this same file - no separate
// vitest.config.js needed. Pure-logic tests only (see AGENT notes in
// src/**/*.test.js) - jsdom is configured because a few of them touch
// localStorage, not because any of them render components.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    // jsdom's localStorage is origin-scoped and stays undefined without a
    // real (non "about:blank") URL - needed for KpiCards.test.js's
    // loadOrder/saveOrder tests, confirmed by direct inspection (typeof
    // window.localStorage was "undefined" until this was added).
    environmentOptions: { jsdom: { url: 'http://localhost' } },
    include: ['src/**/*.test.js'],
  },
})
