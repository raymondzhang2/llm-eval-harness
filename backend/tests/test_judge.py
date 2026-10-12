import json
from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest

from app.schemas import Criterion
from app.services.judge import (
    AnthropicJudge,
    JudgeItem,
    JudgeParseError,
    build_user_prompt,
    parse_judge_response,
)

CRITERION = Criterion(
    name="Clarity",
    description="Is the message easy to understand?",
    anchors={1: "Confusing", 3: "OK", 5: "Crystal clear"},
)
ITEM = JudgeItem(input="Remind user to claim reward", output="Claim your gift now!")


def reply(text: str, stop_reason: str = "end_turn", in_tokens: int = 10, out_tokens: int = 5):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=in_tokens, output_tokens=out_tokens),
    )


def good(score: int = 4) -> str:
    return json.dumps({"rationale": "Short and direct.", "score": score})


class FakeClient:
    """Stands in for AsyncAnthropic; replays scripted replies (or raises scripted errors)."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


async def run(*script):
    client = FakeClient(*script)
    result = await AnthropicJudge(client=client, model="test-model").score(CRITERION, ITEM)
    return result, client


async def test_valid_json_scores_on_first_attempt():
    result, client = await run(reply(good(4)))
    assert result.ok and result.score == 4
    assert result.rationale == "Short and direct."
    assert result.tokens == 15
    assert len(client.calls) == 1


async def test_code_fenced_json_is_accepted():
    result, _ = await run(reply(f"```json\n{good(3)}\n```"))
    assert result.score == 3


async def test_malformed_json_retries_then_succeeds():
    result, client = await run(reply("Sure! The score is 4."), reply(good(4)))
    assert result.score == 4
    assert len(client.calls) == 2
    assert result.tokens == 30  # tokens from both attempts are counted


async def test_out_of_range_score_retries_then_succeeds():
    result, client = await run(reply(good(9)), reply(good(2)))
    assert result.score == 2
    assert len(client.calls) == 2


async def test_gives_up_after_two_retries_and_marks_failed():
    result, client = await run(reply("nope"), reply(good(9)), reply('{"score": 3}'))
    assert not result.ok and result.score is None
    assert len(client.calls) == 3  # 1 attempt + 2 retries
    assert "rationale" in result.error
    assert result.raw_response == '{"score": 3}'  # last raw reply is kept for debugging


async def test_refusal_fails_without_retrying():
    result, client = await run(reply("", stop_reason="refusal"))
    assert not result.ok and "refused" in result.error
    assert len(client.calls) == 1


async def test_api_error_fails_the_judgment_instead_of_raising():
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    result, client = await run(err)
    assert not result.ok and "APIConnectionError" in result.error
    assert len(client.calls) == 1


async def test_request_shape():
    _, client = await run(reply(good()))
    call = client.calls[0]
    assert call["model"] == "test-model"
    assert "temperature" not in call  # rejected by current Claude models
    assert call["messages"][0]["role"] == "user"
    assert call["system"]


def test_prompt_contains_criterion_anchors_and_content():
    prompt = build_user_prompt(CRITERION, ITEM)
    assert "Clarity" in prompt and "1: Confusing" in prompt and "5: Crystal clear" in prompt
    assert "Claim your gift now!" in prompt
    assert "<reference>" not in prompt


def test_prompt_includes_reference_only_when_present():
    item = JudgeItem(input="i", output="o", reference="gold answer")
    assert "<reference>\ngold answer\n</reference>" in build_user_prompt(CRITERION, item)


@pytest.mark.parametrize(
    "text",
    ["", "not json", "[1, 2]", '{"score": 3}', '{"rationale": "x", "score": "high"}'],
)
def test_parse_rejects_bad_shapes(text):
    with pytest.raises(JudgeParseError):
        parse_judge_response(text, CRITERION)
