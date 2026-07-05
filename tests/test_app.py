"""App-boundary tests — no Ollama needed (these fail before any upstream call)."""

from starlette.testclient import TestClient

from toolrails.app import create_app


def _client():
    return TestClient(create_app("http://localhost:11434"))


def test_health():
    with _client() as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_non_object_body_is_rejected_not_crashed():
    # A JSON body that isn't an object must 400, not 500 — the proxy can never
    # be the thing that crashes on weird input.
    with _client() as c:
        for payload in ("null", "[1, 2, 3]", '"hello"', "5"):
            r = c.post("/v1/chat/completions", content=payload,
                       headers={"content-type": "application/json"})
            assert r.status_code == 400, f"{payload} -> {r.status_code}"


def test_invalid_json_body_is_rejected():
    with _client() as c:
        r = c.post("/v1/chat/completions", content="{not json",
                   headers={"content-type": "application/json"})
        assert r.status_code == 400
