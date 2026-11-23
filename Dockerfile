FROM node:22-bookworm-slim
WORKDIR /app
COPY package.json tsconfig.json ./
COPY src ./src
COPY tests ./tests
COPY benchmarks ./benchmarks
RUN npm test && npm run bench
USER node
EXPOSE 8095
CMD ["node", "-e", "import('./src/index.mjs').then(m => m.createStudioServer().listen(8095, '0.0.0.0'))"]
