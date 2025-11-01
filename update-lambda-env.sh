#!/bin/bash
# Lambda 환경변수 즉시 업데이트 스크립트

AWS_REGION="ap-southeast-2"
LAMBDA_FUNCTION_NAME="chatbot-backend-lambda"

echo "Updating Lambda function environment variables..."

aws lambda update-function-configuration \
  --function-name $LAMBDA_FUNCTION_NAME \
  --region $AWS_REGION \
  --environment "Variables={
    ENVIRONMENT=production,
    LOG_LEVEL=INFO,
    ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://3.27.26.7,
    HF_HOME=/tmp/.cache/huggingface,
    TRANSFORMERS_CACHE=/tmp/.cache/huggingface,
    SENTENCE_TRANSFORMERS_HOME=/tmp/.cache/sentence-transformers,
    PINECONE_API_KEY_PARAM=/chatbot/prod/PINECONE_API_KEY,
    PINECONE_INDEX_NAME_PARAM=/chatbot/prod/PINECONE_INDEX_NAME,
    HYPERCLOVA_API_KEY_PARAM=/chatbot/prod/HYPERCLOVA_API_KEY
  }"

echo "Done! Waiting for update to complete..."
aws lambda wait function-updated-v2 --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION
echo "Lambda function updated successfully!"

