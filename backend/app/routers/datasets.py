from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Dataset, Item
from app.schemas import DatasetOut
from app.services.parsing import DatasetParseError, parse_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetOut, status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    db: Session = Depends(get_db),
):
    raw = await file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(413, "File too large")
    try:
        parsed = parse_dataset(file.filename or "", raw)
    except DatasetParseError as e:
        raise HTTPException(422, str(e)) from e

    dataset = Dataset(
        name=name or file.filename or "dataset",
        items=[
            Item(input=p.input, output=p.output, reference=p.reference, metadata_=p.metadata)
            for p in parsed
        ],
    )
    db.add(dataset)
    db.commit()
    return DatasetOut(
        id=dataset.id, name=dataset.name, created_at=dataset.created_at, item_count=len(parsed)
    )


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(404, "Dataset not found")
    count = db.scalar(select(func.count()).select_from(Item).where(Item.dataset_id == dataset_id))
    return DatasetOut(
        id=dataset.id, name=dataset.name, created_at=dataset.created_at, item_count=count or 0
    )
