FROM node:24-alpine AS dependencies

WORKDIR /app

# npm ci follows package-lock exactly and is reproducible in clean images.
COPY package*.json ./
RUN npm ci


# Development target: Vite serves source code and supports bind-mounted updates.
FROM dependencies AS development

COPY . .

EXPOSE 5173
CMD ["npm", "run", "dev"]


# Build target: generate the optimized static bundle for Nginx.
FROM dependencies AS builder

COPY . .
RUN npm run build


# Production target: serve the static bundle and proxy /api to FastAPI.
FROM nginx:1.29.8-alpine AS production

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=6 \
  CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
