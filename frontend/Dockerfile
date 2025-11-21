# --- 1) Dependencies Stage ---
FROM node:18-alpine AS deps
WORKDIR /app

# Alpine 기본 빌드 종속성 설치 (package 설치 실패 방지)
RUN apk add --no-cache python3 make g++

COPY package*.json ./

# npm ci 실패 방지 → 우선 ci, 실패하면 install
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
RUN npm cache clean --force

# --- 2) Builder Stage ---
FROM node:18-alpine AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

RUN npm run build

# --- 3) Production Stage ---
FROM nginx:alpine AS production

WORKDIR /usr/share/nginx/html

# nginx 기본 캐시 제거
RUN rm -rf /etc/nginx/conf.d/*

# 빌드 결과 복사
COPY --from=builder /app/build .

# SPA 라우팅용 Nginx 설정
COPY <<EOF /etc/nginx/conf.d/default.conf
server {
    listen 3000;
    server_name localhost;

    root /usr/share/nginx/html;
    index index.html;

    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_comp_level 5;
    gzip_types text/plain text/css application/json application/javascript image/svg+xml;

    location /static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    location / {
        try_files \$uri /index.html;
    }
}
EOF

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s \
    CMD wget --spider -q http://localhost:3000/ || exit 1

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]