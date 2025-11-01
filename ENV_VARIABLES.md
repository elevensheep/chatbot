# 환경 변수 설정 가이드

## 🔐 GitHub Secrets (CI/CD 파이프라인용)

### **AWS 관련:**
```
ROLE_ARN: arn:aws:iam::144618663232:role/YOUR_ROLE_NAME
AWS_REGION: ap-southeast-2
AWS_ACCOUNT_ID: 144618663232
```

### **ECR 리포지토리:**
```
ECR_BACKEND_REPOSITORY: chatbot-backend
ECR_FRONTEND_REPOSITORY: chatbot-frontend
ECR_REGISTRY: 144618663232.dkr.ecr.ap-southeast-2.amazonaws.com
```

### **ECS 클러스터:**
```
ECS_CLUSTER_NAME: chatbot-ec2-cluster
ECS_SERVICE_NAME: chatbot-service
ECS_TASK_DEFINITION: app-task-definition
```

### **컨테이너 이름:**
```
CONTAINER_BACKEND_NAME: backend-container
CONTAINER_FRONTEND_NAME: frontend-container
```

### **RDS 데이터베이스 (필요시):**
```
DB_HOST: your-rds-endpoint.ap-southeast-2.rds.amazonaws.com
DB_PORT: 5432
DB_NAME: chatbot_db
DB_USER: chatbot_user
DB_PASSWORD: your-secure-password
```

### **ElastiCache Redis (필요시):**
```
REDIS_HOST: your-redis-endpoint.cache.amazonaws.com
REDIS_PORT: 6379
REDIS_PASSWORD: your-redis-password
```

### **Application Load Balancer:**
```
ALB_DNS_NAME: your-alb-dns-name.ap-southeast-2.elb.amazonaws.com
ALB_TARGET_GROUP_ARN: arn:aws:elasticloadbalancing:ap-southeast-2:144618663232:targetgroup/chatbot-tg/xxx
```

## 🚀 애플리케이션 환경 변수 (.env 파일)

### **기본 애플리케이션 설정:**
```bash
# 애플리케이션 정보
APP_NAME=Chatbot API
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# 서버 설정
BACKEND_HOST=0.0.0.0
BACKEND_PORT=5000
BACKEND_WORKERS=1

# 프론트엔드 설정
REACT_APP_API_URL=http://localhost:5000
REACT_APP_ENVIRONMENT=development
NODE_ENV=development
```

### **보안 설정:**
```bash
# JWT 및 세션
SECRET_KEY=your-super-secret-key-change-this-in-production
JWT_SECRET_KEY=your-jwt-secret-key
SESSION_SECRET=your-session-secret

# CORS 설정
# PROD_HOST는 Parameter Store에서 자동으로 가져와 ALLOWED_ORIGINS에 추가됨
# Parameter Store 파라미터: /chatbot/prod/prod_host
PROD_HOST_PARAM=/chatbot/prod/prod_host  # Parameter Store에서 가져올 파라미터 이름 (기본값)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001,https://yourdomain.com
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### **외부 API 설정:**
```bash
# HyperCLOVA X API
HYPERCLOVA_API_KEY=your-hyperclova-api-key-here
HYPERCLOVA_API_GATEWAY_KEY=
HYPERCLOVA_REQUEST_ID=

# OpenAI API
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=1000

# Milvus 벡터 데이터베이스
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=root
MILVUS_PASSWORD=Milvus

# LangChain 설정
LANGCHAIN_API_KEY=your-langchain-api-key
LANGCHAIN_TRACING_V2=true
```

### **데이터베이스 설정:**
```bash
# PostgreSQL
DATABASE_URL=postgresql://username:password@localhost:5432/chatbot_db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=chatbot_db
DB_USER=chatbot_user
DB_PASSWORD=your-password
DB_SSL_MODE=disable

# Redis
REDIS_URL=redis://localhost:6379
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
```

### **파일 저장소:**
```bash
# AWS S3
AWS_S3_BUCKET=chatbot-files
AWS_S3_REGION=ap-southeast-2
AWS_S3_ACCESS_KEY_ID=your-access-key
AWS_S3_SECRET_ACCESS_KEY=your-secret-key

# 로컬 파일 저장
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=10485760  # 10MB
```

### **이메일 설정:**
```bash
# SMTP 설정
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=your-email@gmail.com
```

### **모니터링 및 로깅:**
```bash
# Sentry
SENTRY_DSN=your-sentry-dsn

# DataDog
DD_API_KEY=your-datadog-api-key
DD_SERVICE=chatbot-api
DD_ENV=production

# Prometheus
ENABLE_METRICS=true
METRICS_PORT=9090
```

### **캐싱 설정:**
```bash
# Redis 캐싱
CACHE_TTL=3600  # 1시간
CACHE_PREFIX=chatbot:
ENABLE_CACHE=true
```

### **Rate Limiting:**
```bash
# API 요청 제한
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_BURST=200
```

## 🌐 프로덕션 환경 변수

### **Docker Compose 환경:**
```bash
# 프로덕션 환경
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# 보안 강화
SECRET_KEY=your-super-secure-production-key
JWT_SECRET_KEY=your-production-jwt-key

# 데이터베이스 (프로덕션)
DATABASE_URL=postgresql://prod_user:secure_password@prod-db:5432/chatbot_prod
DB_SSL_MODE=require

# Redis (프로덕션)
REDIS_URL=redis://prod-redis:6379
REDIS_PASSWORD=secure-redis-password

# 외부 API (프로덕션)
OPENAI_API_KEY=your-production-openai-key
```

## 🔧 환경별 설정 파일

### **개발 환경 (.env.development):**
```bash
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://localhost:5432/chatbot_dev
```

### **테스트 환경 (.env.test):**
```bash
ENVIRONMENT=test
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=postgresql://localhost:5432/chatbot_test
```

### **프로덕션 환경 (.env.production):**
```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
DATABASE_URL=postgresql://prod_user:password@prod-db:5432/chatbot_prod
```

## 📋 설정 체크리스트

### **GitHub Secrets:**
- [ ] ROLE_ARN
- [ ] AWS_REGION
- [ ] ECR_BACKEND_REPOSITORY
- [ ] ECR_FRONTEND_REPOSITORY
- [ ] ECS_CLUSTER_NAME
- [ ] ECS_SERVICE_NAME
- [ ] ECS_TASK_DEFINITION
- [ ] CONTAINER_BACKEND_NAME
- [ ] CONTAINER_FRONTEND_NAME

### **애플리케이션 환경 변수:**
- [ ] SECRET_KEY
- [ ] DATABASE_URL
- [ ] REDIS_URL
- [ ] OPENAI_API_KEY
- [ ] MILVUS_HOST
- [ ] ALLOWED_ORIGINS
- [ ] LOG_LEVEL
- [ ] ENVIRONMENT

### **프로덕션 보안:**
- [ ] 강력한 SECRET_KEY 설정
- [ ] 데이터베이스 SSL 연결
- [ ] HTTPS 설정
- [ ] CORS 정책 설정
- [ ] Rate Limiting 설정
