"""적재(ingest)와 질의응답(ask).

답변은 검색된 조항만 근거로 삼는다. 인용할 수 있는 id를 스키마의 enum으로 묶어
모델이 검색되지 않은 조항이나 없는 조항을 지어내 인용할 수 없게 한다.
근거가 없으면 answerable=false로 답을 거절한다.
"""

import re
from dataclasses import dataclass

from .chunk import Chunk
from .store import Hit

SYSTEM = """당신은 회사 규정 안내 도우미입니다.
- 아래 [규정 발췌]에 적힌 내용만 근거로 한국어로 답합니다. 일반 상식이나 추측으로 보태지 않습니다.
- 숫자(일수·금액·기한)는 발췌에 적힌 그대로 씁니다.
- 질문과 관련된 조항을 모두 종합해 답합니다. 한 조항이 "○○ 규정에 따라"처럼 다른 규정을 따르라고 하면, 발췌에 있는 그 규정의 구체적인 내용(기한·연락처·절차)도 함께 적고 인용합니다.
- 질문에 답할 근거가 발췌에 없으면 answerable을 false로, answer를 "규정에서 찾을 수 없습니다."로, citations를 []로 둡니다.
- 답할 수 있으면 answerable을 true로 두고, 답의 근거가 된 조항 id만 citations에 넣습니다."""


@dataclass(frozen=True)
class Answer:
    answerable: bool
    text: str
    citations: list[str]
    hits: list[Hit]


def ingest(chunks: list[Chunk], llm, store) -> int:
    store.reset()
    store.add(chunks, llm.embed([c.text for c in chunks]))
    return store.count()


def answer_schema(ids: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["answerable", "answer", "citations"],
        "properties": {
            "answerable": {"type": "boolean"},
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string", "enum": ids}},
        },
    }


REFERENCE = re.compile(r"(\S+)\s?규정에 따라")


def follow_references(hits: list[Hit], vector: list[float], store) -> list[Hit]:
    """검색된 조항이 "정보보안 규정에 따라"처럼 다른 규정을 가리키면, 그 규정 안에서
    질문과 가장 가까운 조항 하나를 덧붙인다. 질문 표현이 한 규정 쪽으로 치우쳐
    다른 규정의 조항이 상위 k개에서 밀려나는 경우를 메운다."""
    seen = {h.chunk.id for h in hits}
    docs = {h.chunk.doc for h in hits}
    extra = []
    for h in hits:
        for name in REFERENCE.findall(h.chunk.body):
            doc = f"{name}규정"
            if doc in docs:
                continue
            docs.add(doc)
            found = store.search(vector, 1, doc=doc)
            if found and found[0].chunk.id not in seen:
                seen.add(found[0].chunk.id)
                extra.append(found[0])
    return hits + extra


def ask(question: str, llm, store, k: int = 4) -> Answer:
    [vector] = llm.embed([question])
    hits = follow_references(store.search(vector, k), vector, store)
    context = "\n\n".join(f"<{h.chunk.id}>\n{h.chunk.text}" for h in hits)
    ids = [h.chunk.id for h in hits]

    out = llm.complete_json(SYSTEM, f"[규정 발췌]\n{context}\n\n[질문]\n{question}", "policy_answer", answer_schema(ids))

    # enum으로 묶었어도 한 번 더 거른다(중복 제거, 검색 결과 순서 유지)
    citations = [i for i in ids if i in out["citations"]]
    if not out["answerable"]:
        citations = []
    return Answer(out["answerable"], out["answer"], citations, hits)
