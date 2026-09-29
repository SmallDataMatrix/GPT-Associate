"""Background material: typed documents, text extraction, chunking, keyword retrieval and the fixed prompt block."""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

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

# Document kinds, in the order they are laid out for the prep pack. Past transcripts come last: they are
# the noisiest material, so they are distilled into the pack and searched, never pasted into answers raw.
DOC_KINDS = {
    "jd": "Job description / meeting goal",
    "resume": "Resume / about me",
    "prep": "My prep notes / prepared answers",
    "company": "Company / interviewer info",
    "other": "Other",
    "interview": "Past interview / meeting transcript",
}
PACK_VERSION = "2"  # bump when the prep-pack prompt changes, so saved packs are rewritten once
PACK_MIN_TOKENS = 300  # profile fields alone smaller than this (no documents) are used as they are

# Filename hints per kind, checked in this order. ASCII hints match whole words, Chinese ones substrings.
KIND_NAME_HINTS = (
    ("prep", ("prep", "notes", "answers", "stories", "star", "准备", "笔记", "问答")),
    ("interview", ("interview", "transcript", "recording", "meeting", "round", "面试", "录音", "逐字稿", "会议")),
    ("jd", ("jd", "job", "description", "posting", "岗位", "职位", "招聘")),
    ("resume", ("resume", "cv", "简历", "履历")),
    ("company", ("company", "research", "interviewers", "公司", "面试官")),
)
JD_CUES = ("responsibilities", "qualifications", "requirements", "what you'll do", "what you will do",
           "about the role", "preferred", "岗位职责", "任职要求", "职位描述", "任职资格")
RESUME_CUES = ("education", "experience", "skills", "university", "bachelor", "master",
               "教育", "工作经历", "项目经历", "技能")
# "Name: words" or "[00:12] Name: words", and "Speaker 1  00:03" on a line of its own (Otter, Feishu, Zoom).
SPEAKER_LINE = re.compile(
    r"^(?:\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*)?(?P<a>[^\s:：\d][^:：]{0,24}?)\s*[:：]\s*\S"
    r"|^(?P<b>[^\s\d][^:：]{0,24}?)\s+\(?\d{1,2}:\d{2}(?::\d{2})?\)?\s*$"
)
NOT_SPEAKERS = {  # labels that repeat in notes and resumes but are not people
    "q", "a", "question", "answer", "follow-up", "note", "tip", "example", "问", "答",
    "situation", "task", "action", "result", "company", "role", "title", "dates", "location", "skills",
}
CJK = "㐀-䶿一-鿿"


class Doc(NamedTuple):
    name: str
    text: str
    kind: str = "other"


def normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.replace("\r", "").split("\n")]
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


def looks_like_transcript(text: str) -> bool:
    """A few speaker labels that repeat on many lines, as in an exported recording transcript."""
    lines = [line for line in text.splitlines() if line.strip()]
    labels = Counter()
    for line in lines:
        match = SPEAKER_LINE.match(line.strip())
        label = match and (match.group("a") or match.group("b")).strip().lower()
        if label and label not in NOT_SPEAKERS:
            labels[label] += 1
    top = sum(n for _, n in labels.most_common(3))
    return top >= 8 and top >= 0.3 * len(lines)


def guess_kind(name: str, text: str) -> str:
    """Best guess of a document's kind from its name and content; the user can change it."""
    lower_name = name.lower()
    if lower_name.endswith((".srt", ".vtt")) or looks_like_transcript(text):
        return "interview"
    words = set(re.findall(r"[a-z]+", lower_name))
    for kind, hints in KIND_NAME_HINTS:
        if any(h in words if h.isascii() else h in lower_name for h in hints):
            return kind
    head = text[:20000].lower()
    if sum(cue in head for cue in JD_CUES) >= 2:
        return "jd"
    if sum(cue in head for cue in RESUME_CUES) >= 3:
        return "resume"
    return "other"


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
    """English words plus Chinese character bigrams, so Chinese notes are searchable too."""
    words = re.findall(r"[a-z0-9][a-z0-9+#]*(?:\.[a-z0-9]+)*", text.lower())
    tokens = [w for w in words if len(w) > 1 and w not in STOPWORDS]
    for run in re.findall(f"[{CJK}]+", text):
        tokens.extend(run[i : i + 2] for i in range(max(1, len(run) - 1)))
    return tokens


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
    tokens: int = 0  # estimated size of all source material
    pack: bool = False  # the prep pack is part of static_text
    full: bool = False  # every reference document (all but past transcripts) is in static_text verbatim
    index: BM25 | None = None  # past transcripts, and reference documents too large to include in full
    searchable: int = 0  # documents in the index
    terms: list[str] = field(default_factory=list)  # names and jargon that help transcription

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        return self.index.search(query, k) if self.index else []

    def describe(self) -> str:
        parts = []
        if self.pack:
            parts.append("prep pack")
        if self.full:
            parts.append("full text")
        if self.index:
            parts.append(f"search over {self.searchable} doc{'' if self.searchable == 1 else 's'}")
        if parts:
            return " + ".join(parts)
        return "profile fields only" if self.static_text else "nothing added yet"


def kind_of(doc: Doc) -> str:
    return doc.kind if doc.kind in DOC_KINDS else "other"


def profile_header(profile: dict) -> str:
    parts = []
    for field, title in PROFILE_SECTIONS:
        value = (profile.get(field) or "").strip()
        if value:
            parts.append(f"## {title}\n\n{value}")
    return "\n\n".join(parts)


def docs_text(docs: list[Doc]) -> str:
    """Documents grouped by kind, each under its own heading."""
    groups = []
    for kind, title in DOC_KINDS.items():
        body = "\n\n".join(f"### {d.name}\n\n{d.text.strip()}" for d in docs if kind_of(d) == kind and d.text.strip())
        if body:
            groups.append(f"## Documents: {title}\n\n{body}")
    return "\n\n".join(groups)


def source_material(profile: dict, docs: list[Doc]) -> str:
    return "\n\n".join(p for p in (profile_header(profile), docs_text(docs)) if p)


def source_hash(profile: dict, docs: list[Doc]) -> str:
    material = f"{PACK_VERSION}\n{source_material(profile, docs)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def wants_pack(profile: dict, docs: list[Doc]) -> bool:
    """Worth writing a prep pack: there are documents, or the profile fields hold real material (e.g. a pasted JD)."""
    return any(d.text.strip() for d in docs) or estimate_tokens(profile_header(profile)) >= PACK_MIN_TOKENS


def pack_material(profile: dict, docs: list[Doc], max_tokens: int) -> str:
    """Source material for the prep pack, every document trimmed by the same ratio when it is too large."""
    budget = max(max_tokens * 4 - len(profile_header(profile)), 4000)
    total = sum(len(d.text) for d in docs)
    if total > budget:
        docs = [d._replace(text=d.text[: len(d.text) * budget // total]) for d in docs]
    return source_material(profile, docs)


def transcription_terms(profile: dict, docs: list[Doc], max_terms: int = 40) -> list[str]:
    """The user's keywords plus names and jargon that recur in the background."""
    terms = [t.strip() for t in re.split(r"[,;\n]", profile.get("keywords") or "") if t.strip()]
    text = "\n".join([profile.get("about_me") or "", profile.get("goal") or "", profile.get("other_side") or ""])
    text += "\n" + "\n".join(d.text for d in docs)
    counts = Counter(re.findall(r"\b(?:[A-Z][a-z]*[A-Z0-9+#][A-Za-z0-9+#]*|[A-Z][a-z]{2,}|[A-Z]{2,}[0-9]*)\b", text))
    seen = {t.lower() for t in terms}
    for word, count in counts.most_common():
        if len(terms) >= max_terms:
            break
        if count >= 2 and word.lower() not in STOPWORDS and word.lower() not in seen:
            terms.append(word)
            seen.add(word.lower())
    return terms[:max_terms]


def build_knowledge(profile: dict, docs: list[Doc], limit: int, pack: str = "") -> KnowledgeBase:
    """Static block = profile fields + prep pack + reference documents in full when they fit.

    Past transcripts never go in verbatim: the pack distills them and live answers search them.
    """
    docs = [d for d in docs if d.text.strip()]
    terms = transcription_terms(profile, docs)
    header = profile_header(profile)
    if not header and not docs:
        return KnowledgeBase(terms=terms)
    reference = docs_text([d for d in docs if kind_of(d) != "interview"])
    full = bool(reference) and estimate_tokens(reference) <= limit
    pack = pack.strip()
    static = "\n\n".join(p for p in (header, pack and f"## Prep pack\n\n{pack}", full and reference) if p)
    searchable = [d for d in docs if kind_of(d) == "interview" or not full]
    index = None
    if searchable:
        index = BM25([f"[{DOC_KINDS[kind_of(d)]}: {d.name}]\n{c}" for d in searchable for c in chunk_text(d.text)])
    return KnowledgeBase(
        static_text=static,
        tokens=estimate_tokens(source_material(profile, docs)),
        pack=bool(pack),
        full=full,
        index=index,
        searchable=len(searchable),
        terms=terms,
    )
