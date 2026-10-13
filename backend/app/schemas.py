from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    created_at: datetime
    item_count: int


class Criterion(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1)
    scale_min: int = 1
    scale_max: int = 5
    anchors: dict[int, str] = Field(default_factory=dict)  # score -> what it looks like

    @model_validator(mode="after")
    def _check_scale(self):
        if self.scale_min >= self.scale_max:
            raise ValueError("scale_min must be less than scale_max")
        bad = [s for s in self.anchors if not self.scale_min <= s <= self.scale_max]
        if bad:
            raise ValueError(f"anchor scores outside the scale: {sorted(bad)}")
        return self


class RubricCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    criteria: list[Criterion] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self):
        names = [c.name for c in self.criteria]
        if len(set(names)) != len(names):
            raise ValueError("criterion names must be unique")
        return self


class RubricOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    criteria: list[Criterion]
    created_at: datetime


class RunCreate(BaseModel):
    dataset_id: int
    rubric_id: int
    judge_model: str | None = None  # defaults to the server's configured judge model


class RunOut(BaseModel):
    id: int
    dataset_id: int
    rubric_id: int
    judge_model: str
    prompt_version: str
    status: str  # queued, running, done, failed
    total_tokens: int
    created_at: datetime
    judgments_done: int
    judgments_total: int  # items x criteria


class ReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    human_score: int
    note: str | None
    created_at: datetime


class JudgmentOut(BaseModel):
    id: int
    run_id: int
    item_id: int
    criterion: str
    score: int | None  # null when the judgment failed
    rationale: str | None
    error: str | None
    latency_ms: int | None
    tokens: int | None
    input: str
    output: str
    reference: str | None
    review: ReviewOut | None


class JudgmentPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[JudgmentOut]
