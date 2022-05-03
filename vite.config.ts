import path from 'path';
import { defineConfig } from 'vite';
import dts from 'vite-plugin-dts';

const resolvePath = (str: string) => path.resolve(__dirname, str);

export default defineConfig({
  build: {
    lib: {
      entry: resolvePath('lib/index.ts'),
      name: 'Firearm',
      formats: ['es'],
      fileName: (format) => `index.${format}.js`
    },
  },
  worker: {
    format: 'es'
  },
  plugins: [dts()]
});