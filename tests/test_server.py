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
        info = client.post(f"/api/profiles/{slug}/prepare").json()
        assert info["mode"] == "full" and info["slug"] == slug
        assert client.get("/api/profiles").json()["active"] == slug
        assert "Kafka" in client.app.state.meeting.kb.static_text
        assert client.get("/api/profiles/..%2Fsecrets").status_code == 404


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
