from sqlalchemy import func, select

from app.models import Judgment, Run
from app.services.judge import JudgeResult
from app.services.worker import execute_run

CSV = b"input,output,reference\na,out-a,ref-a\nb,out-b,\nc,out-c,\n"
RUBRIC = {
    "name": "R",
    "criteria": [
        {"name": "Clarity", "description": "clear?"},
        {"name": "Tone", "description": "tone?"},
    ],
}


def setup_run(client, csv=CSV, rubric=RUBRIC):
    ds = client.post("/datasets", files={"file": ("d.csv", csv)}).json()
    rb = client.post("/rubrics", json=rubric).json()
    return client.post("/runs", json={"dataset_id": ds["id"], "rubric_id": rb["id"]})


# --- API ---------------------------------------------------------------------


def test_start_run_returns_immediately_and_completes(client, judge):
    r = setup_run(client)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued" and body["judgments_total"] == 6
    assert body["prompt_version"] == "v1" and body["judge_model"]

    got = client.get(f"/runs/{body['id']}").json()  # background task ran after the response
    assert got["status"] == "done"
    assert got["judgments_done"] == got["judgments_total"] == 6
    assert got["total_tokens"] == 60
    assert judge.calls == 6


def test_start_run_with_unknown_dataset_or_rubric(client):
    rb = client.post("/rubrics", json=RUBRIC).json()
    assert client.post("/runs", json={"dataset_id": 99, "rubric_id": rb["id"]}).status_code == 404
    ds = client.post("/datasets", files={"file": ("d.csv", CSV)}).json()
    assert client.post("/runs", json={"dataset_id": ds["id"], "rubric_id": 99}).status_code == 404
    assert client.post("/runs", json={"dataset_id": ds["id"]}).status_code == 422


def test_get_missing_run(client):
    assert client.get("/runs/999").status_code == 404
    assert client.get("/runs/999/judgments").status_code == 404


def test_judgments_include_item_text_and_pagination(client):
    run_id = setup_run(client).json()["id"]
    page = client.get(f"/runs/{run_id}/judgments", params={"limit": 4}).json()
    assert page["total"] == 6 and len(page["items"]) == 4 and page["limit"] == 4
    first = page["items"][0]
    assert first["input"] == "a" and first["output"] == "out-a" and first["reference"] == "ref-a"
    assert first["review"] is None and first["score"] == 3

    rest = client.get(f"/runs/{run_id}/judgments", params={"limit": 4, "offset": 4}).json()
    assert len(rest["items"]) == 2
    assert client.get(f"/runs/{run_id}/judgments", params={"limit": 0}).status_code == 422


def test_judgments_filter_by_criterion_and_score(client, judge):
    judge.behavior = lambda c, i: JudgeResult(
        score=5 if i.output == "out-a" else 2, rationale="x", tokens=1
    )
    run_id = setup_run(client).json()["id"]
    url = f"/runs/{run_id}/judgments"
    assert client.get(url, params={"criterion": "Tone"}).json()["total"] == 3
    assert client.get(url, params={"score": 5}).json()["total"] == 2  # item a x 2 criteria
    both = client.get(url, params={"criterion": "Clarity", "score": 2}).json()
    assert both["total"] == 2


def test_failed_judgments_are_stored_and_filterable(client, judge):
    def flaky(criterion, item):
        if item.output == "out-b":
            return JudgeResult(score=None, raw_response="garbage", tokens=7, error="not valid JSON")
        return JudgeResult(score=4, rationale="ok", tokens=3)

    judge.behavior = flaky
    body = client.get(f"/runs/{setup_run(client).json()['id']}").json()
    assert body["status"] == "done"  # per-item failures do not fail the run
    assert body["judgments_done"] == 6

    failed = client.get(f"/runs/{body['id']}/judgments", params={"failed": True}).json()
    assert failed["total"] == 2
    assert {j["error"] for j in failed["items"]} == {"not valid JSON"}
    assert all(j["score"] is None for j in failed["items"])
    assert (
        client.get(f"/runs/{body['id']}/judgments", params={"failed": False}).json()["total"] == 4
    )


# --- worker ------------------------------------------------------------------


async def test_concurrency_is_limited_by_the_semaphore(client, session_factory, judge):
    csv = b"input,output\n" + b"".join(b"i%d,o%d\n" % (n, n) for n in range(20))
    run_id = setup_run(client, csv).json()["id"]
    with session_factory() as db:
        db.query(Judgment).delete()
        db.get(Run, run_id).status = "queued"
        db.commit()
    judge.calls = judge.max_in_flight = 0

    await execute_run(run_id, session_factory, judge, concurrency=3)
    assert judge.calls == 40
    assert 1 < judge.max_in_flight <= 3


async def test_rerun_resumes_without_duplicating_judgments(client, session_factory, judge):
    run_id = setup_run(client).json()["id"]
    with session_factory() as db:
        for j in db.scalars(select(Judgment).where(Judgment.id.in_([1, 2]))).all():
            db.delete(j)
        db.get(Run, run_id).status = "failed"
        db.commit()
    judge.calls = 0

    await execute_run(run_id, session_factory, judge)
    assert judge.calls == 2  # only the missing pairs are re-judged
    with session_factory() as db:
        assert db.scalar(select(func.count()).select_from(Judgment)) == 6
        assert db.get(Run, run_id).status == "done"


async def test_unexpected_crash_marks_run_failed(client, session_factory, judge):
    run_id = setup_run(client).json()["id"]
    with session_factory() as db:
        db.query(Judgment).delete()
        db.commit()

    def boom(criterion, item):
        raise RuntimeError("kaboom")

    judge.behavior = boom
    await execute_run(run_id, session_factory, judge)
    with session_factory() as db:
        assert db.get(Run, run_id).status == "failed"
