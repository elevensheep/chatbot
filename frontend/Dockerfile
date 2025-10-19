# 멀티스테이지 빌드로 최적화
FROM node:18-alpine AS builder

# 작업 디렉토리 설정
WORKDIR /app

# package.json과 package-lock.json 복사
COPY package*.json ./

# 의존성 설치 (dev 의존성 포함)
RUN npm ci

# 소스 코드 복사
COPY . .

# 환경변수 설정 (빌드 타임)
# 브라우저에서 접근 가능한 localhost 주소 사용
ENV REACT_APP_API_URL=http://localhost:5000

# 프로덕션 빌드
RUN npm run build

# 프로덕션 이미지
FROM nginx:alpine AS production

# 비root 사용자 생성
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# nginx 설정 파일 복사
COPY --from=builder /app/build /usr/share/nginx/html

# nginx 설정 최적화
RUN echo 'server { \
    listen 3000; \
    server_name localhost; \
    root /usr/share/nginx/html; \
    index index.html; \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
    location /static/ { \
        expires 1y; \
        add_header Cache-Control "public, immutable"; \
    } \
}' > /etc/nginx/conf.d/default.conf

# 포트 노출
EXPOSE 3000

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

# nginx 실행
CMD ["nginx", "-g", "daemon off;"]
