"""JSON persistence under data/: background profiles, their documents, and saved meetings."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .knowledge import DOC_KINDS, Doc, guess_kind

PROFILE_FIELDS = ("name", "about_me", "goal", "other_side", "keywords", "base", "brief", "brief_hash")
SESSION_ID = re.compile(r"[\w-]{1,64}")
MAX_SESSIONS_LISTED = 50


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:48] or "default"


def safe_filename(name: str) -> str:
    base = re.sub(r"[^\w.\- ]+", "_", Path(name).name).strip(" .")
    return base[:120] or "document"


def valid_session_id(session_id: str) -> bool:
    return bool(SESSION_ID.fullmatch(session_id or ""))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _join(*texts: str, sep: str = "\n\n") -> str:
    return sep.join(t.strip() for t in texts if t and t.strip())


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.profiles_dir = root / "profiles"
        self.sessions_dir = root / "sessions"

    # Profiles -----------------------------------------------------------------

    def _profile_dir(self, slug: str) -> Path:
        return self.profiles_dir / slugify(slug)

    def list_profiles(self) -> list[dict]:
        if not self.profiles_dir.exists():
            return []
        profiles = []
        for path in sorted(self.profiles_dir.iterdir()):
            if path.is_dir():
                data = _read_json(path / "profile.json")
                profiles.append({"slug": path.name, "name": data.get("name") or path.name})
        return profiles

    def profile_exists(self, slug: str) -> bool:
        return (self._profile_dir(slug) / "profile.json").exists()

    def load_profile(self, slug: str) -> dict:
        data = _read_json(self._profile_dir(slug) / "profile.json")
        profile = {field: str(data.get(field, "")) for field in PROFILE_FIELDS}
        profile["name"] = profile["name"] or slug
        return profile

    def save_profile(self, slug: str, fields: dict) -> dict:
        profile = self.load_profile(slug)
        profile.update({k: str(v) for k, v in fields.items() if k in PROFILE_FIELDS and v is not None})
        _write_json(self._profile_dir(slug) / "profile.json", profile)
        return profile

    def base_of(self, slug: str) -> str:
        """The profile this one builds on, if it is set and still exists (one level only)."""
        base = self.load_profile(slug).get("base", "")
        return base if base and base != slugify(slug) and self.profile_exists(base) else ""

    def load_material(self, slug: str) -> tuple[dict, list[Doc]]:
        """The profile and documents to prepare, with the base profile's About me, keywords and documents merged in."""
        profile = self.load_profile(slug)
        docs = self.load_docs(slug)
        base = self.base_of(slug)
        if not base:
            return profile, docs
        parent = self.load_profile(base)
        profile["about_me"] = _join(parent["about_me"], profile["about_me"])
        profile["keywords"] = _join(parent["keywords"], profile["keywords"], sep=", ")
        inherited = [d._replace(name=f"{parent['name']} / {d.name}") for d in self.load_docs(base)]
        return profile, inherited + docs

    # Documents ----------------------------------------------------------------

    def _docs_dir(self, slug: str) -> Path:
        return self._profile_dir(slug) / "docs"

    def _kinds_path(self, slug: str) -> Path:
        return self._profile_dir(slug) / "docs.json"

    def load_docs(self, slug: str) -> list[Doc]:
        docs_dir = self._docs_dir(slug)
        if not docs_dir.exists():
            return []
        kinds = _read_json(self._kinds_path(slug))
        docs = []
        for path in sorted(docs_dir.glob("*.txt")):
            name = path.name[: -len(".txt")]
            text = path.read_text(encoding="utf-8")
            kind = kinds.get(name)
            docs.append(Doc(name, text, kind if kind in DOC_KINDS else guess_kind(name, text)))
        return docs

    def list_docs(self, slug: str) -> list[dict]:
        return [{"name": d.name, "chars": len(d.text), "kind": d.kind} for d in self.load_docs(slug)]

    def save_doc(self, slug: str, filename: str, text: str, kind: str = "") -> str:
        name = safe_filename(filename)
        path = self._docs_dir(slug) / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.set_doc_kind(slug, name, kind if kind in DOC_KINDS else guess_kind(name, text))
        return name

    def set_doc_kind(self, slug: str, name: str, kind: str) -> bool:
        name = safe_filename(name)
        if kind not in DOC_KINDS or not (self._docs_dir(slug) / f"{name}.txt").exists():
            return False
        kinds = _read_json(self._kinds_path(slug))
        kinds[name] = kind
        _write_json(self._kinds_path(slug), kinds)
        return True

    def delete_doc(self, slug: str, name: str) -> bool:
        name = safe_filename(name)
        path = self._docs_dir(slug) / f"{name}.txt"
        if not path.exists():
            return False
        path.unlink()
        kinds = _read_json(self._kinds_path(slug))
        if kinds.pop(name, None) is not None:
            _write_json(self._kinds_path(slug), kinds)
        return True

    def get_active(self) -> str | None:
        return _read_json(self.root / "state.json").get("active_profile")

    def set_active(self, slug: str) -> None:
        state = _read_json(self.root / "state.json")
        state["active_profile"] = slug
        _write_json(self.root / "state.json", state)

    # Meetings -----------------------------------------------------------------

    def save_session(self, session_id: str, data: dict) -> None:
        _write_json(self.sessions_dir / f"{session_id}.json", data)

    def load_session(self, session_id: str) -> dict:
        return _read_json(self.sessions_dir / f"{session_id}.json") if valid_session_id(session_id) else {}

    def list_sessions(self) -> list[dict]:
        """Saved meetings, newest first, skipping ones where nothing happened."""
        if not self.sessions_dir.exists():
            return []
        sessions = []
        for path in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            data = _read_json(path)
            segments = len(data.get("transcript") or [])
            questions = sum(1 for c in data.get("cards") or [] if c.get("kind") in ("answer", "detail"))
            if segments or questions:
                sessions.append({
                    "id": path.stem,
                    "started": data.get("started") or 0,
                    "profile": (data.get("profile") or {}).get("name") or "",
                    "segments": segments,
                    "questions": questions,
                })
            if len(sessions) >= MAX_SESSIONS_LISTED:
                break
        return sessions

    def load_summary(self, session_id: str) -> dict:
        return _read_json(self.sessions_dir / "summaries" / f"{session_id}.json") if valid_session_id(session_id) else {}

    def save_summary(self, session_id: str, data: dict) -> None:
        _write_json(self.sessions_dir / "summaries" / f"{session_id}.json", data)
