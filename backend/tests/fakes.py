import asyncio

from app.schemas import Criterion
from app.services.judge import Judge, JudgeItem, JudgeResult


class FakeJudge(Judge):
    """Scripted judge: `behavior(criterion, item)` returns a JudgeResult or raises."""

    def __init__(self, behavior=None):
        self.model = "fake-judge"
        self.behavior = behavior or self._default
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    @staticmethod
    def _default(criterion: Criterion, item: JudgeItem) -> JudgeResult:
        score = min(3, criterion.scale_max)
        return JudgeResult(score=score, rationale="fine", raw_response="{}", tokens=10)

    async def score(self, criterion: Criterion, item: JudgeItem) -> JudgeResult:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)  # yield so concurrent calls can overlap
            return self.behavior(criterion, item)
        finally:
            self.in_flight -= 1
