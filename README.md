

# API呼び出し例
curl -X POST "http://localhost:7860/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      "こんにちは、元気ですか？",
      [],
      "あなたは親切で知識豊富な日本語アシスタントです。",
      512,
      0.7,
      0.95
    ]
  }'

