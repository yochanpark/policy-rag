"""python -m rag ingest  — 규정 문서를 나눠 임베딩하고 Chroma에 저장
python -m rag ask "질문" — 규정을 근거로 답하고 출처 조항을 보여 준다"""

import sys

from .chunk import load_policies
from .config import POLICY_DIR, from_env
from .pipeline import ask, ingest


def main(argv: list[str]) -> None:
    if len(argv) < 1 or argv[0] not in ("ingest", "ask") or (argv[0] == "ask" and len(argv) < 2):
        raise SystemExit(__doc__)
    llm, store = from_env()

    if argv[0] == "ingest":
        print(f"조항 {ingest(load_policies(POLICY_DIR), llm, store)}개를 저장했습니다.")
        return

    result = ask(" ".join(argv[1:]), llm, store)
    print(result.text)
    if result.citations:
        print("\n출처: " + ", ".join(result.citations))


if __name__ == "__main__":
    main(sys.argv[1:])
