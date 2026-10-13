from collections.abc import Callable

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.deps import get_judge_factory, get_session_factory
from app.models import Dataset, Item, Judgment, Review, Rubric, Run
from app.schemas import JudgmentOut, JudgmentPage, ReviewOut, RunCreate, RunOut
from app.services.judge import PROMPT_VERSION, Judge
from app.services.worker import execute_run

router = APIRouter(prefix="/runs", tags=["runs"])


def _run_out(db: Session, run: Run) -> RunOut:
    done = db.scalar(select(func.count()).select_from(Judgment).where(Judgment.run_id == run.id))
    n_items = db.scalar(
        select(func.count()).select_from(Item).where(Item.dataset_id == run.dataset_id)
    )
    n_criteria = len(db.get(Rubric, run.rubric_id).criteria)
    return RunOut(
        id=run.id,
        dataset_id=run.dataset_id,
        rubric_id=run.rubric_id,
        judge_model=run.judge_model,
        prompt_version=run.prompt_version,
        status=run.status,
        total_tokens=run.total_tokens,
        created_at=run.created_at,
        judgments_done=done or 0,
        judgments_total=(n_items or 0) * n_criteria,
    )


def _get_run_or_404(db: Session, run_id: int) -> Run:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run


@router.post("", response_model=RunOut, status_code=202)
def start_run(
    body: RunCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    session_factory: Callable[[], Session] = Depends(get_session_factory),
    judge_factory: Callable[[str], Judge] = Depends(get_judge_factory),
):
    """Create the run and return immediately; judging continues in the background."""
    if db.get(Dataset, body.dataset_id) is None:
        raise HTTPException(404, "Dataset not found")
    if db.get(Rubric, body.rubric_id) is None:
        raise HTTPException(404, "Rubric not found")

    run = Run(
        dataset_id=body.dataset_id,
        rubric_id=body.rubric_id,
        judge_model=body.judge_model or settings.judge_model,
        prompt_version=PROMPT_VERSION,
        status="queued",
    )
    db.add(run)
    db.commit()

    background_tasks.add_task(execute_run, run.id, session_factory, judge_factory(run.judge_model))
    return _run_out(db, run)


@router.get("/{run_id}", response_model=RunOut)
def get_run(run_id: int, db: Session = Depends(get_db)):
    return _run_out(db, _get_run_or_404(db, run_id))


@router.get("/{run_id}/judgments", response_model=JudgmentPage)
def list_judgments(
    run_id: int,
    criterion: str | None = None,
    score: int | None = None,
    failed: bool | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    _get_run_or_404(db, run_id)
    filters = [Judgment.run_id == run_id]
    if criterion is not None:
        filters.append(Judgment.criterion == criterion)
    if score is not None:
        filters.append(Judgment.score == score)
    if failed is not None:
        filters.append(Judgment.score.is_(None) if failed else Judgment.score.is_not(None))

    total = db.scalar(select(func.count()).select_from(Judgment).where(*filters))
    rows = db.execute(
        select(Judgment, Item, Review)
        .join(Item, Item.id == Judgment.item_id)
        .outerjoin(Review, Review.judgment_id == Judgment.id)
        .where(*filters)
        .order_by(Judgment.item_id, Judgment.criterion)
        .limit(limit)
        .offset(offset)
    ).all()
    return JudgmentPage(
        total=total or 0,
        limit=limit,
        offset=offset,
        items=[
            JudgmentOut(
                id=j.id,
                run_id=j.run_id,
                item_id=j.item_id,
                criterion=j.criterion,
                score=j.score,
                rationale=j.rationale,
                error=j.error,
                latency_ms=j.latency_ms,
                tokens=j.tokens,
                input=item.input,
                output=item.output,
                reference=item.reference,
                review=ReviewOut.model_validate(r) if r else None,
            )
            for j, item, r in rows
        ],
    )
