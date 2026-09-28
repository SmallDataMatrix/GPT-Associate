"""JSON persistence under data/: background profiles, their documents, and saved meetings."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

PROFILE_FIELDS = ("name", "about_me", "goal", "other_side", "keywords", "brief", "brief_hash")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:48] or "default"


def safe_filename(name: str) -> str:
    base = re.sub(r"[^\w.\- ]+", "_", Path(name).name).strip(" .")
    return base[:120] or "document"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


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

    def list_docs(self, slug: str) -> list[dict]:
        docs_dir = self._profile_dir(slug) / "docs"
        if not docs_dir.exists():
            return []
        return [
            {"name": path.name[: -len(".txt")], "chars": path.stat().st_size}
            for path in sorted(docs_dir.glob("*.txt"))
        ]

    def load_docs(self, slug: str) -> list[tuple[str, str]]:
        docs_dir = self._profile_dir(slug) / "docs"
        if not docs_dir.exists():
            return []
        return [
            (path.name[: -len(".txt")], path.read_text(encoding="utf-8"))
            for path in sorted(docs_dir.glob("*.txt"))
        ]

    def save_doc(self, slug: str, filename: str, text: str) -> str:
        name = safe_filename(filename)
        path = self._profile_dir(slug) / "docs" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return name

    def delete_doc(self, slug: str, name: str) -> bool:
        path = self._profile_dir(slug) / "docs" / f"{safe_filename(name)}.txt"
        if path.exists():
            path.unlink()
            return True
        return False

    def get_active(self) -> str | None:
        return _read_json(self.root / "state.json").get("active_profile")

    def set_active(self, slug: str) -> None:
        state = _read_json(self.root / "state.json")
        state["active_profile"] = slug
        _write_json(self.root / "state.json", state)

    # Meetings -----------------------------------------------------------------

    def save_session(self, session_id: str, data: dict) -> None:
        _write_json(self.sessions_dir / f"{session_id}.json", data)
