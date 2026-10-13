"""Overridable dependencies (tests swap in fakes so no API key or network is needed)."""

from collections.abc import Callable

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.services.judge import AnthropicJudge, Judge


def get_session_factory() -> Callable[[], Session]:
    return SessionLocal


def get_judge_factory() -> Callable[[str], Judge]:
    return lambda model: AnthropicJudge(model=model)
