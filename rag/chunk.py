"""규정 문서를 조항(제N조) 단위로 나눈다.

규정은 조항 하나가 하나의 규칙이라, 글자 수로 자르면 조항이 중간에 끊겨 근거를 인용할 수 없다.
조항 단위로 나누고 id를 "문서 제N조"로 두면 답변의 출처가 곧 사람이 찾아볼 수 있는 위치가 된다.
"""

import re
from dataclasses import dataclass
from pathlib import Path

ARTICLE = re.compile(r"^## (제\d+조)\s*\((.+?)\)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    id: str  # "근태휴가규정 제5조"
    title: str  # "근태 및 휴가 규정 제5조 (연차 휴가 사용)"
    body: str

    @property
    def doc(self) -> str:
        return self.id.split()[0]

    @property
    def text(self) -> str:
        """임베딩과 답변 근거로 쓰는 본문. 조항 본문만으로는 어느 규정인지 모르니 제목을 붙인다."""
        return f"[{self.title}]\n{self.body}"


def chunk_markdown(markdown: str, doc_id: str) -> list[Chunk]:
    doc_title = re.search(r"^# (.+)$", markdown, re.MULTILINE).group(1).strip()
    matches = list(ARTICLE.finditer(markdown))
    chunks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        number, name = m.group(1), m.group(2)
        chunks.append(
            Chunk(
                id=f"{doc_id} {number}",
                title=f"{doc_title} {number} ({name})",
                body=markdown[m.end() : end].strip(),
            )
        )
    return chunks


def load_policies(directory: Path) -> list[Chunk]:
    chunks = []
    for path in sorted(directory.glob("*.md")):
        doc_id = path.stem.split("-", 1)[-1]  # "01-근태휴가규정" → "근태휴가규정"
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), doc_id))
    return chunks
