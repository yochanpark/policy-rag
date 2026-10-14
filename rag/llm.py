"""임베딩·답변 생성 클라이언트.

Gemini의 OpenAI 호환 주소를 openai SDK로 부른다. 주소·모델 이름만 바꾸면 OpenAI나
Azure OpenAI로 그대로 옮길 수 있다. 429·5xx는 SDK가 지수 백오프로 다시 시도한다.
"""

import json

from openai import OpenAI

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
EMBED_BATCH = 100  # Gemini 임베딩 요청 한 번에 넣을 수 있는 최대 개수


class LlmClient:
    def __init__(self, api_key: str, embed_model: str, chat_model: str, base_url: str = GEMINI_BASE_URL):
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=5)
        self.embed_model = embed_model
        self.chat_model = chat_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for i in range(0, len(texts), EMBED_BATCH):
            response = self._client.embeddings.create(model=self.embed_model, input=texts[i : i + EMBED_BATCH])
            vectors.extend(d.embedding for d in response.data)
        return vectors

    def complete_json(self, system: str, user: str, schema_name: str, schema: dict) -> dict:
        response = self._client.chat.completions.create(
            model=self.chat_model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
        )
        return json.loads(response.choices[0].message.content)
