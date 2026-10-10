from fastapi import FastAPI

from app.routers import datasets, rubrics

app = FastAPI(title="LLM Eval Harness")
app.include_router(datasets.router)
app.include_router(rubrics.router)


@app.get("/health")
def health():
    return {"status": "ok"}
