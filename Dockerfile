FROM node:20-slim

# Dépendances système pour sharp et onnxruntime-node
RUN apt-get update && apt-get install -y \
    python3 \
    make \
    g++ \
    libvips-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier les manifestes en premier (cache Docker)
COPY package*.json ./
COPY tsconfig.json ./

# Installer toutes les dépendances (y compris devDependencies pour le build)
RUN npm install

# Copier les sources et builder
COPY src/ ./src/
RUN npm run build

# Nettoyer les devDependencies
RUN npm prune --production

EXPOSE 3001

CMD ["node", "dist/index.js"]
