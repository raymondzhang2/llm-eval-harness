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
