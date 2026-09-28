"""Background material: text extraction, chunking, keyword retrieval and the fixed prompt block."""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from dataclasses import dataclass, field

STOPWORDS = frozenset(
    """a an and are as at be been but by for from had has have i in is it its of on or that the this
    to was were will with you your we our they their he she his her them me my not can do does did so if
    what how why when where who which would could should about into than then there these those also just
    like more most some any all one our us an very""".split()
)

PROFILE_SECTIONS = (
    ("about_me", "About me"),
    ("goal", "Meeting goal / job description"),
    ("other_side", "About the other side"),
    ("keywords", "Key terms"),
)


def normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.replace("\r", "").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def extract_text(filename: str, data: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == "docx":
        import docx

        document = docx.Document(io.BytesIO(data))
        parts = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        text = "\n".join(parts)
    else:
        text = data.decode("utf-8-sig", errors="replace")
    return normalize_whitespace(text)


def estimate_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def chunk_text(text: str, size: int = 800) -> list[str]:
    pieces = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        while len(para) > size:
            cut = para.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            pieces.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            pieces.append(para)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > size:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9+#]*(?:\.[a-z0-9]+)*", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


class BM25:
    """Small in-memory BM25 keyword index: instant and free, good enough for resumes and notes."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        tokens = [tokenize(d) for d in docs]
        self.lengths = [len(t) for t in tokens]
        self.avgdl = sum(self.lengths) / len(docs) if docs else 0.0
        self.tf = [Counter(t) for t in tokens]
        df = Counter(term for t in tokens for term in set(t))
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[str]:
        terms = set(tokenize(query))
        scored = []
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1))
            score = sum(
                self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf
            )
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [self.docs[i] for _, i in scored[:k]]


@dataclass
class KnowledgeBase:
    static_text: str = ""  # background block that stays fixed for the whole meeting (prompt-cached)
    mode: str = "empty"  # empty | full | brief | indexed
    tokens: int = 0  # estimated size of all source material
    index: BM25 | None = None  # only when the material is too large to include in full
    terms: list[str] = field(default_factory=list)  # names and jargon that help transcription

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        return self.index.search(query, k) if self.index else []


def profile_header(profile: dict) -> str:
    parts = []
    for field, title in PROFILE_SECTIONS:
        value = (profile.get(field) or "").strip()
        if value:
            parts.append(f"## {title}\n\n{value}")
    return "\n\n".join(parts)


def docs_text(docs: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"### {name}\n\n{text.strip()}" for name, text in docs if text.strip())


def source_hash(profile: dict, docs: list[tuple[str, str]]) -> str:
    material = profile_header(profile) + "\n\n" + docs_text(docs)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def source_material(profile: dict, docs: list[tuple[str, str]]) -> str:
    body = docs_text(docs)
    return "\n\n".join(p for p in (profile_header(profile), f"## Documents\n\n{body}" if body else "") if p)


def needs_brief(profile: dict, docs: list[tuple[str, str]], limit: int) -> bool:
    return estimate_tokens(source_material(profile, docs)) > limit


def transcription_terms(profile: dict, docs: list[tuple[str, str]], max_terms: int = 40) -> list[str]:
    """The user's keywords plus names and jargon that recur in the background."""
    terms = [t.strip() for t in re.split(r"[,;\n]", profile.get("keywords") or "") if t.strip()]
    text = "\n".join([profile.get("about_me") or "", profile.get("goal") or "", profile.get("other_side") or ""])
    text += "\n" + "\n".join(t for _, t in docs)
    counts = Counter(re.findall(r"\b(?:[A-Z][a-z]*[A-Z0-9+#][A-Za-z0-9+#]*|[A-Z][a-z]{2,}|[A-Z]{2,}[0-9]*)\b", text))
    seen = {t.lower() for t in terms}
    for word, count in counts.most_common():
        if len(terms) >= max_terms:
            break
        if count >= 2 and word.lower() not in STOPWORDS and word.lower() not in seen:
            terms.append(word)
            seen.add(word.lower())
    return terms[:max_terms]


def build_knowledge(profile: dict, docs: list[tuple[str, str]], limit: int, brief: str = "") -> KnowledgeBase:
    material = source_material(profile, docs)
    terms = transcription_terms(profile, docs)
    if not material:
        return KnowledgeBase(terms=terms)
    tokens = estimate_tokens(material)
    if tokens <= limit:
        return KnowledgeBase(static_text=material, mode="full", tokens=tokens, terms=terms)
    index = BM25([f"[{name}]\n{chunk}" for name, text in docs for chunk in chunk_text(text)])
    header = profile_header(profile)
    if brief.strip():
        static = "\n\n".join(p for p in (header, f"## Background brief\n\n{brief.strip()}") if p)
        return KnowledgeBase(static_text=static, mode="brief", tokens=tokens, index=index, terms=terms)
    return KnowledgeBase(static_text=header, mode="indexed", tokens=tokens, index=index, terms=terms)
