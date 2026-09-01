FROM node:24-slim
WORKDIR /app

# npm ci follows package-lock exactly and is reproducible in clean images.
COPY package*.json ./
RUN npm ci

# Copy source after dependencies to preserve Docker layer caching.
COPY . .

EXPOSE 5173
CMD ["npm", "run", "dev"]
