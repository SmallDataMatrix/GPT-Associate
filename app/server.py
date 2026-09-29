"""FastAPI app: the page, profile/upload APIs, the live UI socket and the audio socket."""

from __future__ import annotations

import asyncio
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.requests import HTTPConnection
from starlette.websockets import WebSocketDisconnect

from .answerer import Answerer
from .config import STATIC_DIR, Settings
from .knowledge import DOC_KINDS, extract_text, source_hash
from .meeting import MeetingSession
from .store import Store, slugify, valid_session_id
from .transcriber import Transcriber

COOKIE = "ga_code"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
AUDIO_SOURCES = ("them", "me", "room")
PROFILE_INPUTS = ("name", "about_me", "goal", "other_side", "keywords", "base", "brief")
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}

LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>GPT Associate</title>
<style>body{font-family:system-ui,sans-serif;background:#0f1115;color:#e8eaed;display:grid;place-items:center;
min-height:100vh;margin:0}form{display:flex;flex-direction:column;gap:12px;padding:24px;width:min(320px,90vw)}
input,button{font-size:20px;padding:12px;border-radius:10px;border:1px solid #3a3f4b;background:#1a1d24;color:inherit}
button{background:#3b82f6;border:0;color:#fff}</style></head><body>
<form method="get" action="/"><label for="c">Access code (shown in the terminal)</label>
<input id="c" name="code" inputmode="numeric" autocomplete="one-time-code" autofocus>
<button>Open</button></form></body></html>"""


def create_app(settings: Settings, client=None, transcriber_factory=Transcriber) -> FastAPI:
    store = Store(settings.data_dir)
    meeting = MeetingSession(settings, store, Answerer(client, settings), transcriber_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active = store.get_active()
        if active and store.profile_exists(active):
            meeting.load_profile(active)
        ticker = asyncio.create_task(meeting.run_ticker())
        yield
        ticker.cancel()
        await meeting.shutdown()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.meeting = meeting
    app.state.store = store

    def authorized(conn: HTTPConnection) -> bool:
        host = conn.client.host if conn.client else ""
        if host in LOCAL_HOSTS or host.startswith("::ffff:127."):
            return True
        code = settings.access_code
        return secrets.compare_digest(conn.cookies.get(COOKIE, ""), code) or secrets.compare_digest(
            conn.query_params.get("code", ""), code
        )

    @app.middleware("http")
    async def access_code(request: Request, call_next):
        # Static files are just the page's code; data lives behind /api and the sockets.
        if not request.url.path.startswith("/static/") and not authorized(request):
            if request.url.path == "/":
                return HTMLResponse(LOGIN_PAGE, status_code=401)
            return JSONResponse({"error": "Access code required."}, status_code=401)
        response = await call_next(request)
        if request.query_params.get("code") == settings.access_code:
            response.set_cookie(COOKIE, settings.access_code, max_age=7 * 86400, httponly=True, samesite="lax")
        return response

    def check_slug(slug: str) -> str:
        if slugify(slug) != slug:
            raise HTTPException(404, "Unknown profile.")
        return slug

    def check_session(session_id: str) -> str:
        if not valid_session_id(session_id):
            raise HTTPException(404, "Unknown meeting.")
        return session_id

    def profile_payload(slug: str) -> dict:
        """A profile for the Background panel: its fields, its documents and the ones it reuses from its base."""
        base = store.base_of(slug)
        base_name = store.load_profile(base)["name"] if base else ""
        base_docs = [{**d, "from": base_name} for d in store.list_docs(base)] if base else []
        return {**store.load_profile(slug), "slug": slug, "docs": store.list_docs(slug),
                "base_docs": base_docs, "kinds": DOC_KINDS}

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/api/profiles")
    async def list_profiles():
        return {"profiles": store.list_profiles(), "active": meeting.profile.get("slug", "")}

    @app.post("/api/profiles")
    async def create_profile(body: dict):
        name = str(body.get("name", "")).strip() or "Default"
        slug = slugify(name)
        if not store.profile_exists(slug):
            store.save_profile(slug, {"name": name})
        return {"slug": slug}

    @app.get("/api/profiles/{slug}")
    async def get_profile(slug: str):
        check_slug(slug)
        return profile_payload(slug)

    @app.put("/api/profiles/{slug}")
    async def save_profile(slug: str, body: dict):
        check_slug(slug)
        before = store.load_profile(slug)
        fields = {k: body[k] for k in PROFILE_INPUTS if k in body}
        base = str(fields.get("base") or "")
        if base and (base == slug or not store.profile_exists(base)):
            raise HTTPException(400, "A profile can only build on another existing profile.")
        store.save_profile(slug, fields)
        if "brief" in body and body["brief"] != before.get("brief"):
            # A hand-edited prep pack is kept until the documents or profile fields change.
            store.save_profile(slug, {"brief_hash": source_hash(*store.load_material(slug))})
        return profile_payload(slug)

    @app.post("/api/profiles/{slug}/docs")
    async def upload_docs(slug: str, files: list[UploadFile]):
        check_slug(slug)
        errors = []
        for upload in files:
            data = await upload.read(MAX_UPLOAD_BYTES + 1)
            name = upload.filename or "document"
            if len(data) > MAX_UPLOAD_BYTES:
                errors.append(f"{name}: larger than 25 MB")
                continue
            try:
                text = await asyncio.to_thread(extract_text, name, data)
            except Exception as exc:
                errors.append(f"{name}: could not read ({exc})")
                continue
            if not text.strip():
                errors.append(f"{name}: no text found (scanned PDFs need OCR first)")
                continue
            store.save_doc(slug, name, text)
        return {"docs": store.list_docs(slug), "errors": errors}

    @app.patch("/api/profiles/{slug}/docs/{name}")
    async def set_doc_kind(slug: str, name: str, body: dict):
        check_slug(slug)
        if not store.set_doc_kind(slug, name, str(body.get("kind", ""))):
            raise HTTPException(400, "Unknown document or type.")
        return {"docs": store.list_docs(slug)}

    @app.delete("/api/profiles/{slug}/docs/{name}")
    async def delete_doc(slug: str, name: str):
        check_slug(slug)
        store.delete_doc(slug, name)
        return {"docs": store.list_docs(slug)}

    @app.post("/api/profiles/{slug}/import")
    async def import_meeting(slug: str, body: dict):
        check_slug(slug)
        if not store.profile_exists(slug):
            raise HTTPException(404, "Unknown profile.")
        if not await meeting.import_meeting(slug, check_session(str(body.get("session_id", "")))):
            raise HTTPException(404, "That meeting has nothing recorded.")
        return profile_payload(slug)

    @app.get("/api/sessions")
    async def list_sessions():
        meeting.save()  # so the meeting in progress is listed too
        return {"sessions": store.list_sessions(), "current": meeting.id}

    @app.get("/api/sessions/{session_id}/summary.md")
    async def meeting_summary(session_id: str):
        text = await meeting.meeting_summary(check_session(session_id))
        if text is None:
            raise HTTPException(404, "Nothing has been recorded in this meeting yet.")
        headers = {"Content-Disposition": f'attachment; filename="meeting-{session_id}.md"'}
        return Response(text, media_type="text/markdown; charset=utf-8", headers=headers)

    @app.post("/api/profiles/{slug}/prepare")
    async def prepare(slug: str):
        check_slug(slug)
        try:
            return await meeting.activate_profile(slug)
        except Exception as exc:
            raise HTTPException(502, f"Preparing the background failed: {exc}") from exc

    @app.websocket("/ws/ui")
    async def ui_socket(ws: WebSocket):
        if not authorized(ws):
            await ws.close(code=1008)
            return
        await ws.accept()
        await meeting.hub.serve(ws, meeting.snapshot(), meeting.command)

    @app.websocket("/ws/audio")
    async def audio_socket(ws: WebSocket):
        source = ws.query_params.get("src", "")
        if not authorized(ws) or source not in AUDIO_SOURCES:
            await ws.close(code=1008)
            return
        await ws.accept()
        transcriber = meeting.attach_audio(source)
        try:
            async for chunk in ws.iter_bytes():
                transcriber.push(chunk)
        except WebSocketDisconnect:
            pass
        finally:
            await meeting.detach_audio(source, transcriber)

    return app
