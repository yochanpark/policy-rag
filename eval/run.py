"""실제 모델로 평가 세트를 돌린다.

python -m eval.run            (먼저 python -m rag ingest)
EVAL_DELAY=4 python -m eval.run  — 무료 등급 분당 한도가 걸리면 질문 사이를 띄운다

세 가지를 따로 잰다.
- 검색: 인용해야 할 조항이 검색 결과(상위 k개) 안에 들어왔는가
- 답변: 인용해야 할 조항을 모두 인용했고, 필요한 사실(숫자 등)이 답에 들어 있는가
- 거절: 규정에 없는 질문은 답하지 않고, 있는 질문은 답하는가
"""

import json
import os
import time
from pathlib import Path

from rag.config import from_env
from rag.pipeline import ask

K = 4


def has_fact(answer: str, fact) -> bool:
    text = answer.replace(" ", "")
    options = fact if isinstance(fact, list) else [fact]
    return any(o.replace(" ", "") in text for o in options)


def main() -> None:
    questions = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))["questions"]
    llm, store = from_env()
    if store.count() == 0:
        raise SystemExit("색인이 비어 있습니다. 먼저 python -m rag ingest를 실행하세요.")
    delay = float(os.environ.get("EVAL_DELAY", "0"))

    retrieved = needed = 0
    answered_ok = answerable_n = 0
    refusal_ok = 0
    passed = 0
    for n, item in enumerate(questions):
        if delay and n:
            time.sleep(delay)
        expect_answer = item.get("answerable", True)
        try:
            result = ask(item["q"], llm, store, k=K)
        except Exception as error:  # 한 문항 실패로 전체를 멈추지 않는다
            print(f"✗ {item['q']}\n    호출 실패: {' '.join(str(error).split())[:200]}")
            continue

        problems = []
        if result.answerable != expect_answer:
            problems.append(f"answerable: 기대 {expect_answer} / 실제 {result.answerable}")
        else:
            refusal_ok += 1

        if expect_answer:
            answerable_n += 1
            hit_ids = {h.chunk.id for h in result.hits}
            retrieved += sum(c in hit_ids for c in item["cite"])
            needed += len(item["cite"])
            missing_search = [c for c in item["cite"] if c not in hit_ids]
            missing_cite = [c for c in item["cite"] if c not in result.citations]
            missing_fact = [f for f in item["facts"] if not has_fact(result.text, f)]
            if missing_search:
                problems.append(f"검색 누락: {missing_search}")
            if missing_cite:
                problems.append(f"인용 누락: {missing_cite} (인용: {result.citations})")
            if missing_fact:
                problems.append(f"사실 누락: {missing_fact}")
            if result.answerable and not missing_cite and not missing_fact:
                answered_ok += 1

        if not problems:
            passed += 1
        mark = "✗" if problems else "✓"
        detail = "".join(f"\n    {p}" for p in problems)
        if problems:
            detail += f"\n    답: {result.text}"
        print(f"{mark} {item['q']}{detail}")

    total = len(questions)
    print(
        f"\n통과 {passed}/{total}"
        f" | 검색 재현율@{K} {retrieved}/{needed}"
        f" | 답변(인용+사실) {answered_ok}/{answerable_n}"
        f" | 답변/거절 판단 {refusal_ok}/{total}"
        f" — {llm.embed_model} + {llm.chat_model}"
    )


if __name__ == "__main__":
    main()
