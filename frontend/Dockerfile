# 멀티스테이지 빌드로 최적화
FROM node:18-alpine AS deps

# 작업 디렉토리 설정
WORKDIR /app

# package.json과 package-lock.json 복사
COPY package*.json ./

# 의존성 설치 최적화 (빌드 시 모든 의존성 필요)
RUN npm ci --no-audit --no-fund && \
    npm cache clean --force && \
    rm -rf /tmp/* /var/cache/apk/*

# 빌드 스테이지
FROM node:18-alpine AS builder

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 복사
COPY --from=deps /app/node_modules ./node_modules

# package.json과 package-lock.json 복사
COPY package*.json ./

# 소스 코드 복사
COPY . .

# 프로덕션 빌드 최적화
RUN npm run build && \
    rm -rf node_modules && \
    rm -rf src && \
    rm -rf public && \
    rm -rf package*.json && \
    rm -rf README.md && \
    rm -rf .git* && \
    rm -rf *.md

# 프로덕션 이미지 (nginx:alpine 최적화)
FROM nginx:alpine AS production

# 불필요한 패키지 제거 및 최적화
RUN apk del --no-cache apk-tools && \
    rm -rf /var/cache/apk/* /tmp/* /var/tmp/* && \
    rm -rf /usr/share/man/* /usr/share/doc/*

# nginx 설정 파일 복사
COPY --from=builder /app/build /usr/share/nginx/html

# 런타임 환경변수 주입 스크립트 복사
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# nginx 설정 최적화 (압축 및 캐싱)
RUN echo 'server { \
    listen 3000; \
    server_name localhost; \
    root /usr/share/nginx/html; \
    index index.html; \
    \
    # Gzip 압축 설정 \
    gzip on; \
    gzip_vary on; \
    gzip_min_length 1024; \
    gzip_comp_level 6; \
    gzip_types \
        text/plain \
        text/css \
        text/xml \
        text/javascript \
        application/javascript \
        application/xml+rss \
        application/json \
        application/xml \
        image/svg+xml; \
    \
    # 정적 파일 캐싱 \
    location /static/ { \
        expires 1y; \
        add_header Cache-Control "public, immutable"; \
        access_log off; \
    } \
    \
    location /assets/ { \
        expires 1y; \
        add_header Cache-Control "public, immutable"; \
        access_log off; \
    } \
    \
    # 메인 라우팅 \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
    \
    # 보안 헤더 \
    add_header X-Frame-Options "SAMEORIGIN" always; \
    add_header X-Content-Type-Options "nosniff" always; \
    add_header X-XSS-Protection "1; mode=block" always; \
}' > /etc/nginx/conf.d/default.conf

# 포트 노출
EXPOSE 3000

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

# entrypoint 스크립트 실행 (환경변수 주입 후 nginx 시작)
CMD ["/entrypoint.sh"]
