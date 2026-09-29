from fastapi.testclient import TestClient

from app.server import create_app

from .conftest import FakeClient, FakeTranscriber


def make_client(settings):
    return TestClient(create_app(settings, FakeClient(), FakeTranscriber))


def test_other_devices_need_the_access_code(settings):
    with make_client(settings) as client:
        assert client.get("/").status_code == 401
        assert client.get("/api/profiles").status_code == 401
        assert client.get("/?code=000000").status_code == 401
        assert client.get("/static/app.js").status_code == 200  # page code only, no data
        ok = client.get("/?code=123456")
        assert ok.status_code == 200 and "GPT Associate" in ok.text
        assert client.get("/api/profiles").status_code == 200  # the cookie now carries the code


def test_profile_upload_and_prepare(settings):
    with make_client(settings) as client:
        client.get("/?code=123456")
        slug = client.post("/api/profiles", json={"name": "Acme Interview"}).json()["slug"]
        assert slug == "acme-interview"
        client.put(f"/api/profiles/{slug}", json={"about_me": "I am Sam.", "goal": "Staff engineer"})
        files = [("files", ("resume.txt", b"Led the Kafka migration at Globex.", "text/plain")),
                 ("files", ("empty.txt", b"   ", "text/plain"))]
        uploaded = client.post(f"/api/profiles/{slug}/docs", files=files).json()
        assert [d["name"] for d in uploaded["docs"]] == ["resume.txt"]
        assert len(uploaded["errors"]) == 1
        assert uploaded["docs"][0]["kind"] == "resume"
        info = client.post(f"/api/profiles/{slug}/prepare").json()
        assert info["summary"] == "prep pack + full text" and info["slug"] == slug
        assert client.get("/api/profiles").json()["active"] == slug
        kb_text = client.app.state.meeting.kb.static_text
        assert "Kafka" in kb_text and "## Prep pack" in kb_text
        assert client.get(f"/api/profiles/{slug}").json()["brief"] == "- Participants: Dana"
        assert client.get("/api/profiles/..%2Fsecrets").status_code == 404


def test_doc_kind_can_be_changed(settings):
    with make_client(settings) as client:
        client.get("/?code=123456")
        slug = client.post("/api/profiles", json={"name": "Acme"}).json()["slug"]
        client.post(f"/api/profiles/{slug}/docs", files=[("files", ("misc.txt", b"Some notes.", "text/plain"))])
        docs = client.patch(f"/api/profiles/{slug}/docs/misc.txt", json={"kind": "prep"}).json()["docs"]
        assert docs == [{"name": "misc.txt", "chars": 11, "kind": "prep"}]
        assert client.patch(f"/api/profiles/{slug}/docs/misc.txt", json={"kind": "bogus"}).status_code == 400
        assert client.patch(f"/api/profiles/{slug}/docs/nope.txt", json={"kind": "prep"}).status_code == 400
        assert client.get(f"/api/profiles/{slug}").json()["kinds"]["interview"].startswith("Past interview")


def test_profile_builds_on_a_base_profile(settings):
    with make_client(settings) as client:
        client.get("/?code=123456")
        base = client.post("/api/profiles", json={"name": "My career"}).json()["slug"]
        job = client.post("/api/profiles", json={"name": "Acme"}).json()["slug"]
        client.put(f"/api/profiles/{base}", json={"about_me": "I am Sam.", "keywords": "Kafka"})
        client.post(f"/api/profiles/{base}/docs", files=[("files", ("resume.txt", b"Led Kafka at Globex.", "text/plain"))])
        assert client.put(f"/api/profiles/{job}", json={"base": job}).status_code == 400
        assert client.put(f"/api/profiles/{job}", json={"base": "missing"}).status_code == 400
        profile = client.put(f"/api/profiles/{job}", json={"base": base, "keywords": "Acme"}).json()
        assert profile["base_docs"] == [{"name": "resume.txt", "chars": 20, "kind": "resume", "from": "My career"}]
        client.post(f"/api/profiles/{job}/prepare")
        kb = client.app.state.meeting.kb
        assert "I am Sam." in kb.static_text and "### My career / resume.txt" in kb.static_text
        assert kb.terms[:2] == ["Kafka", "Acme"]


def test_meetings_are_listed_summarized_and_imported(settings):
    with make_client(settings) as client:
        client.get("/?code=123456")
        slug = client.post("/api/profiles", json={"name": "Acme"}).json()["slug"]
        meeting = client.app.state.meeting
        meeting.item_times["i1"] = {"start": meeting.started + 5, "stop": meeting.started + 7}
        meeting.auto_answer = False
        meeting.on_final("them", "i1", "Why do you want to join Acme?")
        listed = client.get("/api/sessions").json()
        assert listed["current"] == meeting.id
        assert [m["id"] for m in listed["sessions"]] == [meeting.id]
        res = client.get(f"/api/sessions/{meeting.id}/summary.md")
        assert res.status_code == 200 and res.headers["content-type"].startswith("text/markdown")
        assert "attachment" in res.headers["content-disposition"]
        assert "- Participants: Dana" in res.text  # the fake model's summary
        assert "**Them** [00:05]: Why do you want to join Acme?" in res.text
        docs = client.post(f"/api/profiles/{slug}/import", json={"session_id": meeting.id}).json()["docs"]
        assert docs[0]["kind"] == "interview" and docs[0]["name"] == f"meeting-{meeting.id}.md"
        assert client.get("/api/sessions/nope/summary.md").status_code == 404
        assert client.get("/api/sessions/..%2F..%2Fstate/summary.md").status_code == 404
        assert client.post(f"/api/profiles/{slug}/import", json={"session_id": "nope"}).status_code == 404


def test_live_socket_streams_answers(settings):
    with make_client(settings) as client:
        client.get("/?code=123456")
        with client.websocket_connect("/ws/ui?code=123456") as ws:
            assert ws.receive_json()["type"] == "snapshot"
            ws.send_json({"type": "ask", "text": "Tell me about yourself"})
            seen = []
            while True:
                event = ws.receive_json()
                seen.append(event)
                if event["type"] == "card" and event["card"]["status"] == "done":
                    break
            assert event["card"]["text"] == "I led the migration."


def test_audio_socket_feeds_the_transcriber(settings):
    FakeTranscriber.instances.clear()
    with make_client(settings) as client:
        client.get("/?code=123456")
        with client.websocket_connect("/ws/audio?src=them&code=123456") as ws:
            ws.send_bytes(b"\x00\x01" * 1200)
            ws.send_bytes(b"\x00\x01" * 1200)
        transcriber = FakeTranscriber.instances[-1]
        assert transcriber.source == "them" and transcriber.started
        assert len(transcriber.chunks) == 2
        assert transcriber.stopped
