FROM node:24.21.0-alpine3.24@sha256:be80f76cf40ec8e42b9bec49f60a55e0660f30af58d3e5a25530785b30ea67e2 AS dependencies

WORKDIR /app

# npm ci follows package-lock exactly and is reproducible in clean images.
RUN chown node:node /app
COPY --chown=node:node package*.json ./

USER node

RUN npm ci


# Development target: Vite serves source code and supports bind-mounted updates.
FROM dependencies AS development

COPY --chown=node:node . .

EXPOSE 5173
CMD ["npm", "run", "dev"]


# Build target: generate the optimized static bundle for Nginx.
FROM dependencies AS builder

COPY --chown=node:node . .
RUN npm run build


# Production target: serve the static bundle and proxy /api to FastAPI.
FROM nginx:1.29.8-alpine3.23@sha256:5616878291a2eed594aee8db4dade5878cf7edcb475e59193904b198d9b830de AS production

COPY --chown=nginx:nginx nginx.conf /etc/nginx/conf.d/default.conf
COPY --chown=nginx:nginx --from=builder /app/dist /usr/share/nginx/html

# Nginx listens on an unprivileged port, so its master and worker processes can
# both run as the image's nginx user.
RUN sed -i '/^user[[:space:]]/d' /etc/nginx/nginx.conf \
    && sed -i 's#pid[[:space:]]\+/run/nginx.pid;#pid /tmp/nginx.pid;#' /etc/nginx/nginx.conf \
    && chown -R nginx:nginx \
        /etc/nginx/conf.d \
        /usr/share/nginx/html \
        /var/cache/nginx \
        /var/log/nginx \
        /var/run

USER nginx

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=6 \
  CMD wget -q -O /dev/null http://127.0.0.1:8080/ || exit 1
