from rag.chunk import chunk_markdown, load_policies
from rag.config import POLICY_DIR

DOC = """# 근태 및 휴가 규정

> 머리말은 조항이 아니다.

## 제1조 (목적)
목적 본문.

## 제5조 (연차 휴가 사용)
① 3근무일 전까지 신청한다.
② 반차는 0.5일이다.
"""


def test_splits_by_article_and_keeps_numbers():
    chunks = chunk_markdown(DOC, "근태휴가규정")
    assert [c.id for c in chunks] == ["근태휴가규정 제1조", "근태휴가규정 제5조"]
    assert chunks[1].body == "① 3근무일 전까지 신청한다.\n② 반차는 0.5일이다."


def test_text_carries_document_title():
    # 조항 본문만으로는 어느 규정인지 알 수 없다
    chunk = chunk_markdown(DOC, "근태휴가규정")[1]
    assert chunk.text.startswith("[근태 및 휴가 규정 제5조 (연차 휴가 사용)]\n")


def test_preamble_is_not_a_chunk():
    assert all("머리말" not in c.body for c in chunk_markdown(DOC, "근태휴가규정"))


def test_all_policies_load_with_unique_ids():
    chunks = load_policies(POLICY_DIR)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert {i.split()[0] for i in ids} == {"근태휴가규정", "출장여비규정", "정보보안규정", "재택근무규정", "자산관리규정", "복리후생규정"}
    assert all(c.body for c in chunks)
