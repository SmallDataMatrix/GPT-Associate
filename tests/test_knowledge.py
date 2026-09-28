import io

from app.knowledge import (
    BM25,
    build_knowledge,
    chunk_text,
    extract_text,
    needs_brief,
    source_hash,
    transcription_terms,
)

PROFILE = {"name": "Acme", "about_me": "I am Sam, a backend engineer.", "goal": "Senior engineer at Acme", "keywords": "Kubernetes, Acme"}


def test_chunks_respect_size_and_keep_all_text():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(20))
    chunks = chunk_text(text, size=800)
    assert all(len(c) <= 800 for c in chunks)
    joined = " ".join(chunks)
    assert all(f"Paragraph {i} " in joined for i in range(20))


def test_long_paragraph_is_split():
    chunks = chunk_text("x" * 50 + " " + "y" * 2000, size=800)
    assert len(chunks) >= 3
    assert all(len(c) <= 800 for c in chunks)


def test_bm25_finds_the_relevant_chunk():
    docs = [
        "Led the Kubernetes migration for 40 services, cut deploy time by 70%.",
        "Organised the company hackathon and mentored two interns.",
        "Built a Kafka pipeline processing 2 billion events per day.",
    ]
    index = BM25(docs)
    assert index.search("Tell me about your Kafka experience", k=1) == [docs[2]]
    assert index.search("kubernetes migration", k=1) == [docs[0]]
    assert index.search("zebra", k=3) == []


def test_small_material_is_included_in_full():
    kb = build_knowledge(PROFILE, [("resume.txt", "Worked at Globex on billing.")], limit=15000)
    assert kb.mode == "full"
    assert "Worked at Globex on billing." in kb.static_text
    assert "I am Sam" in kb.static_text
    assert kb.index is None


def test_large_material_uses_brief_and_index():
    big = "\n\n".join(f"Project {i}: shipped feature {i} with Postgres and Redis." for i in range(3000))
    docs = [("notes.txt", big)]
    assert needs_brief(PROFILE, docs, limit=15000)
    kb = build_knowledge(PROFILE, docs, limit=15000, brief="Sam is a backend engineer.")
    assert kb.mode == "brief"
    assert "Sam is a backend engineer." in kb.static_text
    assert big not in kb.static_text
    assert kb.retrieve("Project 1234")
    assert build_knowledge(PROFILE, docs, limit=15000).mode == "indexed"


def test_empty_profile():
    kb = build_knowledge({}, [], limit=15000)
    assert kb.mode == "empty" and kb.static_text == ""


def test_terms_include_keywords_and_frequent_names():
    docs = [("resume.txt", "Worked with PostgreSQL at Globex. Globex used PostgreSQL heavily. The end. The start.")]
    terms = transcription_terms(PROFILE, docs)
    assert terms[:2] == ["Kubernetes", "Acme"]
    assert "Globex" in terms and "PostgreSQL" in terms and "The" not in terms


def test_source_hash_changes_with_material():
    a = source_hash(PROFILE, [("a.txt", "one")])
    assert a == source_hash(PROFILE, [("a.txt", "one")])
    assert a != source_hash(PROFILE, [("a.txt", "two")])


def test_extract_txt_and_docx():
    assert extract_text("notes.md", "﻿Hello   world\r\n\r\n\r\n\r\nBye".encode()) == "Hello world\n\nBye"

    import docx

    document = docx.Document()
    document.add_paragraph("Senior engineer at Globex")
    buffer = io.BytesIO()
    document.save(buffer)
    assert "Senior engineer at Globex" in extract_text("cv.docx", buffer.getvalue())
