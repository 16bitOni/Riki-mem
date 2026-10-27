"""The benchmark's QA mode: context() -> fixed reader -> official judge, with costs measured."""

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "longeval"))

from harness import Question, QAClients, aggregate, judge_prompt, score_question  # noqa: E402

from riki_memory import MemoryConfig, MemoryCreate, RikiMemory, OpenAICompatibleProvider  # noqa: E402


def _fake_openai(log):
    def handler(request):
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        log.append(body)
        assert len(body["messages"]) == 1 and body["temperature"] == 0  # user-only, as LongMemEval calls it
        if prompt.startswith("I will give you several facts"):
            reply = "The dog is called Bruno." if "Bruno" in prompt else "I don't know."
        else:  # the judge
            reply = "yes" if "Bruno" in prompt.split("Model Response:")[1] else "no"
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    p = OpenAICompatibleProvider(api_key="k", base_url="http://fake/v1", max_retries=0)
    p._client = httpx.AsyncClient(base_url="http://fake/v1", transport=httpx.MockTransport(handler))
    return p


def test_the_official_judge_prompt_is_chosen_by_question_type():
    assert "off-by-one" in judge_prompt("temporal-reasoning", "q1", "Q", "A", "R")
    assert "updated answer" in judge_prompt("knowledge-update", "q1", "Q", "A", "R")
    assert "Rubric:" in judge_prompt("single-session-preference", "q1", "Q", "A", "R")
    assert "unanswerable" in judge_prompt("multi-session", "q7_abs", "Q", "A", "R")
    assert judge_prompt("multi-session", "q1", "Q", "A", "R").endswith("Answer yes or no only.")


async def test_a_question_is_answered_from_context_and_graded(db_url, provider):
    calls = []
    fake = _fake_openai(calls)
    qa = QAClients(reader=fake, reader_model="reader-x", judge=fake, judge_model="judge-y")
    async with RikiMemory(database_url=db_url, llm=provider, config=MemoryConfig(planner_mode="heuristic")) as mem:
        await mem.save(MemoryCreate(user_id="q1", content="Sub adopted a dog named Bruno.", kind="fact"))
        q = Question(question_id="q1", question="What is my dog's name? Bruno?", answers=["Bruno"],
                     category="single-session-user", question_date="2026/10/01 (Thu) 10:00",
                     evidence_session_ids=[])
        result = await score_question(mem, "q1", q, judge="qa", qa=qa)

    assert result.hit and "Bruno" in result.answer
    assert [c["model"] for c in calls] == ["reader-x", "judge-y"] and calls[1]["max_tokens"] == 10
    assert result.context_tokens > 0 and result.read_model_calls == 0
    report = aggregate([result])["overall"]
    assert report["accuracy"] == 1.0 and report["context_tokens_avg"] == result.context_tokens
