CSV = b"input,output\nhello,Hi there!\nbye,See ya\n"

RUBRIC = {
    "name": "Notifications",
    "criteria": [
        {
            "name": "Clarity",
            "description": "Is the message easy to understand?",
            "anchors": {"1": "Confusing", "3": "OK", "5": "Crystal clear"},
        },
        {"name": "Safety", "description": "Is it safe?", "scale_min": 1, "scale_max": 3},
    ],
}


def upload(client, content=CSV, filename="d.csv", **data):
    return client.post("/datasets", files={"file": (filename, content)}, data=data)


def test_upload_and_get_dataset(client):
    r = upload(client, name="Demo")
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "Demo" and body["item_count"] == 2

    got = client.get(f"/datasets/{body['id']}")
    assert got.status_code == 200
    assert got.json()["item_count"] == 2


def test_upload_defaults_name_to_filename(client):
    assert upload(client).json()["name"] == "d.csv"


def test_upload_rejects_invalid_files(client):
    assert upload(client, b"").status_code == 422
    assert upload(client, b"input,text\na,b\n").status_code == 422
    assert upload(client, b"input,output\n\xff,x\n").status_code == 422
    assert upload(client, CSV, "d.pdf").status_code == 422


def test_invalid_upload_stores_nothing(client):
    upload(client, b"input,output\na,b\nc,\n")
    assert client.get("/datasets/1").status_code == 404


def test_get_missing_dataset(client):
    assert client.get("/datasets/999").status_code == 404


def test_create_rubric(client):
    r = client.post("/rubrics", json=RUBRIC)
    assert r.status_code == 201
    crit = r.json()["criteria"]
    assert crit[0]["scale_min"] == 1 and crit[0]["scale_max"] == 5
    assert crit[0]["anchors"] == {"1": "Confusing", "3": "OK", "5": "Crystal clear"}
    assert crit[1]["scale_max"] == 3


def test_rubric_validation(client):
    assert client.post("/rubrics", json={"name": "x", "criteria": []}).status_code == 422
    dup = {"name": "x", "criteria": [RUBRIC["criteria"][0]] * 2}
    assert client.post("/rubrics", json=dup).status_code == 422
    bad_scale = {"name": "x", "criteria": [{"name": "a", "description": "b", "scale_min": 5}]}
    assert client.post("/rubrics", json=bad_scale).status_code == 422
    bad_anchor = {
        "name": "x",
        "criteria": [{"name": "a", "description": "b", "anchors": {"9": "off scale"}}],
    }
    assert client.post("/rubrics", json=bad_anchor).status_code == 422
