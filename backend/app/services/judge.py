"""LLM-as-judge: one call per item per criterion, returning validated JSON.

The judge asks for {"rationale": ..., "score": n} in that order (reasoning before the
score), validates it with Pydantic against the criterion's scale, and retries malformed
or out-of-range answers. If every attempt fails it returns a failed result instead of
raising, so one bad item never crashes a run.

`Judge` is the provider interface: adding another provider means one new subclass.
"""

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import anthropic
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.schemas import Criterion

PROMPT_VERSION = "v1"
MAX_PARSE_RETRIES = 2  # so up to 3 attempts per judgment

SYSTEM_PROMPT = (
    "You are a strict, consistent grader. You score one output against one criterion on a "
    "fixed scale. Apply the scale anchors literally and do not reward length or style that "
    "the criterion does not ask for. Respond with a single JSON object and nothing else: "
    '{"rationale": "<brief reasoning citing the output>", "score": <integer>}. '
    "Write the rationale first, then the score."
)


@dataclass(frozen=True)
class JudgeItem:
    input: str
    output: str
    reference: str | None = None


@dataclass
class JudgeResult:
    score: int | None  # None when the judgment failed
    rationale: str | None = None
    raw_response: str | None = None
    latency_ms: int | None = None
    tokens: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.score is not None


class JudgeOutput(BaseModel):
    rationale: str
    score: int


class JudgeParseError(ValueError):
    pass


def build_user_prompt(criterion: Criterion, item: JudgeItem) -> str:
    anchors = "\n".join(f"  {score}: {text}" for score, text in sorted(criterion.anchors.items()))
    parts = [
        f"Criterion: {criterion.name}",
        f"Description: {criterion.description}",
        f"Scale: integer from {criterion.scale_min} to {criterion.scale_max}",
    ]
    if anchors:
        parts.append(f"Anchors:\n{anchors}")
    parts.append(f"<input>\n{item.input}\n</input>")
    parts.append(f"<output>\n{item.output}\n</output>")
    if item.reference:
        parts.append(f"<reference>\n{item.reference}\n</reference>")
    return "\n\n".join(parts)


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_judge_response(text: str, criterion: Criterion) -> JudgeOutput:
    """Parse and validate a model reply; raises JudgeParseError with a short reason."""
    cleaned = text.strip()
    if m := _FENCE.match(cleaned):
        cleaned = m.group(1)
    try:
        out = JudgeOutput.model_validate(json.loads(cleaned))
    except json.JSONDecodeError as e:
        raise JudgeParseError("response is not valid JSON") from e
    except ValidationError as e:
        raise JudgeParseError("response is missing rationale/score or has wrong types") from e
    if not criterion.scale_min <= out.score <= criterion.scale_max:
        raise JudgeParseError(
            f"score {out.score} outside scale {criterion.scale_min}-{criterion.scale_max}"
        )
    return out


class Judge(ABC):
    model: str

    @abstractmethod
    async def score(self, criterion: Criterion, item: JudgeItem) -> JudgeResult: ...


class AnthropicJudge(Judge):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None = None,
        model: str | None = None,
        max_tokens: int = 2048,
    ):
        # The SDK itself retries 429/5xx/connection errors with exponential backoff.
        self.client = client or anthropic.AsyncAnthropic(max_retries=5)
        self.model = model or settings.judge_model
        self.max_tokens = max_tokens

    async def score(self, criterion: Criterion, item: JudgeItem) -> JudgeResult:
        prompt = build_user_prompt(criterion, item)
        started = time.monotonic()
        tokens = 0
        raw: str | None = None
        error = "no attempts made"

        for _ in range(1 + MAX_PARSE_RETRIES):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                    output_config={"effort": "low"},
                )
            except anthropic.APIError as e:  # retries are exhausted inside the SDK
                error = f"API error: {type(e).__name__}"
                break

            tokens += response.usage.input_tokens + response.usage.output_tokens
            if response.stop_reason == "refusal":
                error = "model refused to grade this item"
                break  # retrying the same prompt will not change a refusal

            raw = "".join(b.text for b in response.content if b.type == "text")
            try:
                out = parse_judge_response(raw, criterion)
            except JudgeParseError as e:
                error = str(e)
                continue
            return JudgeResult(
                score=out.score,
                rationale=out.rationale,
                raw_response=raw,
                latency_ms=round((time.monotonic() - started) * 1000),
                tokens=tokens,
            )

        return JudgeResult(
            score=None,
            raw_response=raw,
            latency_ms=round((time.monotonic() - started) * 1000),
            tokens=tokens,
            error=error,
        )
