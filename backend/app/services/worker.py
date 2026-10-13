"""Background run worker: judges every (item, criterion) pair with limited concurrency.

Each judgment is committed as soon as it finishes, so the frontend can poll progress.
Pairs that already have a judgment are skipped, so re-running a run resumes it instead of
duplicating work (the table is also unique on run/item/criterion). A judgment that fails
is stored with a null score and an error; only an unexpected crash marks the run failed.
"""

import asyncio
import logging
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Item, Judgment, Rubric, Run
from app.schemas import Criterion
from app.services.judge import Judge, JudgeItem

logger = logging.getLogger(__name__)


async def execute_run(
    run_id: int,
    session_factory: Callable[[], Session],
    judge: Judge,
    concurrency: int | None = None,
) -> None:
    with session_factory() as db:
        try:
            await _execute(db, run_id, judge, concurrency or settings.judge_concurrency)
        except Exception:
            logger.exception("run %s crashed", run_id)
            db.rollback()
            run = db.get(Run, run_id)
            if run is not None:
                run.status = "failed"
                db.commit()


async def _execute(db: Session, run_id: int, judge: Judge, concurrency: int) -> None:
    run = db.get(Run, run_id)
    if run is None:
        return
    rubric = db.get(Rubric, run.rubric_id)
    criteria = [Criterion.model_validate(c) for c in rubric.criteria]
    items = db.scalars(
        select(Item).where(Item.dataset_id == run.dataset_id).order_by(Item.id)
    ).all()
    already = set(
        db.execute(
            select(Judgment.item_id, Judgment.criterion).where(Judgment.run_id == run_id)
        ).all()
    )
    pending = [(i, c) for i in items for c in criteria if (i.id, c.name) not in already]

    run.status = "running"
    db.commit()

    semaphore = asyncio.Semaphore(concurrency)

    async def judge_one(item: Item, criterion: Criterion) -> None:
        async with semaphore:
            result = await judge.score(
                criterion, JudgeItem(input=item.input, output=item.output, reference=item.reference)
            )
        # No awaits below, so each judgment is saved atomically on the event loop.
        db.add(
            Judgment(
                run_id=run_id,
                item_id=item.id,
                criterion=criterion.name,
                score=result.score,
                rationale=result.rationale,
                raw_response=result.raw_response,
                error=result.error,
                latency_ms=result.latency_ms,
                tokens=result.tokens,
            )
        )
        run.total_tokens += result.tokens or 0
        db.commit()

    # TaskGroup cancels the remaining judgments if one raises unexpectedly.
    async with asyncio.TaskGroup() as group:
        for item, criterion in pending:
            group.create_task(judge_one(item, criterion))

    run.status = "done"
    db.commit()
