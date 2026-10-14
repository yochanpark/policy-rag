import hashlib
import math

import pytest

from rag.chunk import Chunk
from rag.pipeline import answer_schema, ask, ingest
from rag.store import ChromaStore

CHUNKS = [
    Chunk("근태휴가규정 제4조", "근태 및 휴가 규정 제4조 (연차 휴가 발생)", "80% 이상 출근하면 연차 15일을 준다."),
    Chunk("출장여비규정 제5조", "출장 및 여비 규정 제5조 (숙박비)", "서울 숙박비 한도는 1박 12만 원이다."),
    Chunk("복리후생규정 제6조", "복리후생 규정 제6조 (휴양소)", "휴양소는 1년에 4박까지 이용한다."),
]


class FakeLlm:
    """글자 2개 묶음을 해시한 벡터로 임베딩을 흉내 낸다. 답변은 미리 정해 둔 값을 돌려준다."""

    def __init__(self, reply=None):
        self.reply = reply
        self.prompts = []

    def embed(self, texts):
        vectors = []
        for t in texts:
            v = [0.0] * 64
            for a, b in zip(t, t[1:]):
                v[hashlib.md5((a + b).encode()).digest()[0] % 64] += 1
            norm = math.sqrt(sum(x * x for x in v)) or 1
            vectors.append([x / norm for x in v])
        return vectors

    def complete_json(self, system, user, schema_name, schema):
        self.prompts.append((user, schema))
        return self.reply(schema) if callable(self.reply) else self.reply


@pytest.fixture
def store(tmp_path):
    return ChromaStore(str(tmp_path / "chroma"))


def test_ingest_replaces_previous_index(store):
    llm = FakeLlm()
    assert ingest(CHUNKS, llm, store) == 3
    assert ingest(CHUNKS[:2], llm, store) == 2  # 다시 적재하면 처음부터


def test_search_finds_the_related_article(store):
    llm = FakeLlm()
    ingest(CHUNKS, llm, store)
    [hit, *_] = store.search(llm.embed(["서울 숙박비 한도"])[0], k=3)
    assert hit.chunk.id == "출장여비규정 제5조"
    assert hit.chunk.title == "출장 및 여비 규정 제5조 (숙박비)"


def test_only_retrieved_ids_can_be_cited(store):
    llm = FakeLlm(reply=lambda schema: {"answerable": True, "answer": "12만 원", "citations": schema["properties"]["citations"]["items"]["enum"][:1]})
    ingest(CHUNKS, llm, store)
    result = ask("서울 숙박비 한도", llm, store, k=2)

    user, schema = llm.prompts[0]
    enum = schema["properties"]["citations"]["items"]["enum"]
    assert enum == [h.chunk.id for h in result.hits]  # 인용 가능한 id = 검색된 조항
    assert "<출장여비규정 제5조>" in user  # 근거 조항이 프롬프트에 들어간다
    assert result.citations == ["출장여비규정 제5조"]


def test_citations_outside_retrieved_are_dropped(store):
    # 스키마를 지키지 않는 모델이 있어도 검색되지 않은 조항은 인용으로 남기지 않는다
    llm = FakeLlm(reply={"answerable": True, "answer": "…", "citations": ["없는 규정 제99조", "출장여비규정 제5조"]})
    ingest(CHUNKS, llm, store)
    assert ask("서울 숙박비 한도", llm, store, k=2).citations == ["출장여비규정 제5조"]


def test_unanswerable_has_no_citations(store):
    llm = FakeLlm(reply={"answerable": False, "answer": "규정에서 찾을 수 없습니다.", "citations": ["출장여비규정 제5조"]})
    ingest(CHUNKS, llm, store)
    result = ask("주차 지원금은?", llm, store)
    assert (result.answerable, result.citations) == (False, [])


def test_follows_reference_to_another_policy(store):
    # 검색된 조항이 "정보보안 규정에 따라"처럼 다른 규정을 가리키면, 그 규정 안에서 다시 찾아 덧붙인다
    chunks = [
        Chunk("자산관리규정 제5조", "업무 자산 관리 규정 제5조 (분실)", "노트북 분실 시 자산 담당자와 정보보안 규정에 따라 정보보안팀에 신고한다."),
        Chunk("정보보안규정 제2조", "정보보안 규정 제2조 (비밀번호)", "비밀번호는 12자 이상으로 한다."),
        Chunk("정보보안규정 제7조", "정보보안 규정 제7조 (사고 신고)", "노트북 분실 등 보안 사고는 1시간 안에 신고한다."),
        Chunk("재택근무규정 제5조", "재택근무 규정 제5조 (준수)", "재택근무 중 노트북은 회사 지급품만 쓴다."),
    ]
    llm = FakeLlm(reply={"answerable": True, "answer": "…", "citations": []})
    ingest(chunks, llm, store)
    result = ask("노트북 분실 시 자산 담당자 신고", llm, store, k=1)

    ids = [h.chunk.id for h in result.hits]
    assert ids[0] == "자산관리규정 제5조"
    assert ids[1:] == ["정보보안규정 제7조"]  # 참조된 규정에서 질문과 가장 가까운 조항


def test_reference_already_retrieved_is_not_added_twice(store):
    chunks = [
        Chunk("자산관리규정 제5조", "업무 자산 관리 규정 제5조 (분실)", "분실 시 정보보안 규정에 따라 신고한다."),
        Chunk("정보보안규정 제7조", "정보보안 규정 제7조 (사고 신고)", "분실 시 1시간 안에 신고한다."),
    ]
    llm = FakeLlm(reply={"answerable": True, "answer": "…", "citations": []})
    ingest(chunks, llm, store)
    ids = [h.chunk.id for h in ask("분실 신고", llm, store, k=2).hits]
    assert sorted(ids) == ["자산관리규정 제5조", "정보보안규정 제7조"]


def test_schema_is_strict():
    schema = answer_schema(["a"])
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["answerable", "answer", "citations"]
