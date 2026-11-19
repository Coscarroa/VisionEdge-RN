import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(() => {
  return {
    plugins: [
      react(),
      tailwindcss(),
      // Copies src/assets into dist/ so root-relative URLs like
      // "/src/assets/images/..." (e.g. generate_image output) still resolve
      // in production builds. Do not remove.
      {
        name: 'copy-unbundled-src-assets',
        apply: 'build',
        closeBundle(error?: Error): void {
          if (error) {
            return;
          }
          const distDir = path.resolve(__dirname, 'dist');
          const srcAssets = path.resolve(__dirname, 'src/assets');
          const distAssets = path.resolve(distDir, 'src/assets');
          if (fs.existsSync(distDir) && fs.existsSync(srcAssets)) {
            fs.cpSync(srcAssets, distAssets, {recursive: true, force: true});
          }
          // Copiar binarios locales .wasm y .mjs de onnxruntime-web a dist/ort-wasm/
          const ortDistDir = path.resolve(__dirname, 'node_modules/onnxruntime-web/dist');
          const targetOrtDir = path.resolve(distDir, 'ort-wasm');
          if (fs.existsSync(distDir) && fs.existsSync(ortDistDir)) {
            fs.mkdirSync(targetOrtDir, {recursive: true});
            for (const file of fs.readdirSync(ortDistDir)) {
              if (file.startsWith('ort-wasm') && (file.endsWith('.wasm') || file.endsWith('.mjs'))) {
                fs.cpSync(path.join(ortDistDir, file), path.join(targetOrtDir, file), {force: true});
              }
            }
          }
        },
      },
      // Middleware para servir binarios locales de ONNX Runtime Web (/ort-wasm/*) y modelos (/models/*.onnx) con MIME types exactos
      {
        name: 'onnx-wasm-mime-headers',
        configureServer(server) {
          const ortDistDir = path.resolve(__dirname, 'node_modules/onnxruntime-web/dist');
          server.middlewares.use((req, res, next) => {
            const cleanUrl = req.url?.split('?')[0] ?? '';

            if (cleanUrl.startsWith('/ort-wasm/')) {
              const fileName = path.basename(cleanUrl);
              const filePath = path.join(ortDistDir, fileName);
              if (fs.existsSync(filePath)) {
                if (fileName.endsWith('.wasm')) {
                  res.setHeader('Content-Type', 'application/wasm');
                } else if (fileName.endsWith('.mjs') || fileName.endsWith('.js')) {
                  res.setHeader('Content-Type', 'text/javascript; charset=utf-8');
                }
                res.setHeader('Access-Control-Allow-Origin', '*');
                res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
                fs.createReadStream(filePath).pipe(res);
                return;
              }
            }

            if (cleanUrl.endsWith('.wasm')) {
              res.setHeader('Content-Type', 'application/wasm');
              res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
            } else if (cleanUrl.endsWith('.onnx')) {
              res.setHeader('Content-Type', 'application/octet-stream');
            }
            res.setHeader('Access-Control-Allow-Origin', '*');
            next();
          });
        },
      },
    ],
    assetsInclude: ['**/*.onnx', '**/*.wasm'],
    optimizeDeps: {
      exclude: ['onnxruntime-web'],
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify—file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
      headers: {
        'Access-Control-Allow-Origin': '*',
      },
    },
  };
});
