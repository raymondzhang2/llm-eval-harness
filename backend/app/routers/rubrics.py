from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Rubric
from app.schemas import RubricCreate, RubricOut

router = APIRouter(prefix="/rubrics", tags=["rubrics"])


@router.post("", response_model=RubricOut, status_code=201)
def create_rubric(body: RubricCreate, db: Session = Depends(get_db)):
    rubric = Rubric(name=body.name, criteria=body.model_dump(mode="json")["criteria"])
    db.add(rubric)
    db.commit()
    return rubric
