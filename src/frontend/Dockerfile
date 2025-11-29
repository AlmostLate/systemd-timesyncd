# ---- Stage 1: Build ----
FROM node:25-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN npm run build

# ---- Stage 2: Run ----
FROM nginx:stable-alpine

RUN addgroup -S appgroup && adduser -S appuser -G appgroup

COPY nginx.conf /etc/nginx/nginx.conf

COPY --from=builder /app/dist /usr/share/nginx/html

RUN chown -R appuser:appgroup /var/cache/nginx /var/log/nginx /etc/nginx/conf.d
RUN touch /var/run/nginx.pid && chown -R appuser:appgroup /var/run/nginx.pid

USER appuser

EXPOSE 8080

CMD [ "nginx", "-g", "daemon off;" ]
