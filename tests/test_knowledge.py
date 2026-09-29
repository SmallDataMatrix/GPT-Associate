import io

from app.knowledge import (
    BM25,
    Doc,
    build_knowledge,
    chunk_text,
    extract_text,
    guess_kind,
    pack_material,
    source_hash,
    transcription_terms,
    wants_pack,
)

PROFILE = {"name": "Acme", "about_me": "I am Sam, a backend engineer.", "goal": "Senior engineer at Acme", "keywords": "Kubernetes, Acme"}

TRANSCRIPT = "\n".join(
    f"Interviewer: Question {i} about Kafka?\nSam: I used Kafka for {i} years." for i in range(6)
)


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


def test_bm25_searches_chinese_text():
    docs = ["我在字节跳动负责推荐系统的实时特征平台。", "我组织过公司的黑客马拉松，也带过两个实习生。"]
    assert BM25(docs).search("说说你做推荐系统的经历", k=1) == [docs[0]]


def test_guess_kind_from_name_and_content():
    assert guess_kind("Acme_JD.pdf", "") == "jd"
    assert guess_kind("Sam CV.docx", "") == "resume"
    assert guess_kind("面试准备.md", "") == "prep"
    assert guess_kind("round1.txt", TRANSCRIPT) == "interview"
    assert guess_kind("otter.txt", "Speaker 1  00:03\nHi\nSpeaker 2  00:07\nHello\n" * 5) == "interview"
    jd = "About the role\nResponsibilities: build pipelines\nQualifications: 5 years of Python"
    assert guess_kind("acme.pdf", jd) == "jd"
    # Q/A and STAR labels repeat in prep notes but are not speakers.
    notes = "\n".join(f"Q: question {i}\nA: answer {i}\nResult: done" for i in range(8))
    assert guess_kind("misc.md", notes) == "other"
    assert guess_kind("misc.md", "Just some text.") == "other"


def test_small_material_is_included_in_full():
    kb = build_knowledge(PROFILE, [Doc("resume.txt", "Worked at Globex on billing.", "resume")], limit=15000)
    assert kb.full and not kb.pack
    assert "Worked at Globex on billing." in kb.static_text
    assert "I am Sam" in kb.static_text
    assert kb.index is None
    assert kb.describe() == "full text"


def test_prep_pack_goes_first_and_transcripts_are_only_searched():
    docs = [Doc("jd.txt", "Senior engineer, Kafka required.", "jd"), Doc("round1.txt", TRANSCRIPT, "interview")]
    kb = build_knowledge(PROFILE, docs, limit=15000, pack="Q: Why Kafka?\nA: It fits our scale.")
    assert kb.static_text.index("## Prep pack") < kb.static_text.index("Kafka required")
    assert "Question 3 about Kafka" not in kb.static_text
    assert any("Question 3 about Kafka" in e for e in kb.retrieve("question 3 about kafka"))
    assert kb.retrieve("kafka")[0].startswith("[Past interview / meeting transcript: round1.txt]")
    assert kb.describe() == "prep pack + full text + search over 1 doc"


def test_large_material_is_searched():
    big = "\n\n".join(f"Project {i}: shipped feature {i} with Postgres and Redis." for i in range(3000))
    docs = [Doc("notes.txt", big, "prep")]
    kb = build_knowledge(PROFILE, docs, limit=15000, pack="Sam is a backend engineer.")
    assert "Sam is a backend engineer." in kb.static_text
    assert big not in kb.static_text and not kb.full
    assert kb.retrieve("Project 1234")
    assert build_knowledge(PROFILE, docs, limit=15000).describe() == "search over 1 doc"


def test_empty_profile():
    kb = build_knowledge({}, [], limit=15000)
    assert kb.static_text == "" and kb.describe() == "nothing added yet"
    assert build_knowledge(PROFILE, [], limit=15000).describe() == "profile fields only"


def test_pack_is_wanted_for_documents_or_long_profile_fields():
    assert wants_pack(PROFILE, [Doc("a.txt", "one")])
    assert not wants_pack(PROFILE, [])
    assert wants_pack({**PROFILE, "goal": "Requirements: " + "Python " * 300}, [])


def test_pack_material_trims_every_document_evenly():
    docs = [Doc("a.txt", "a" * 40000, "resume"), Doc("b.txt", "b" * 120000, "interview")]
    material = pack_material({}, docs, max_tokens=10000)
    assert material.count("a") < 40000 and material.count("b") < 120000
    assert 2.5 < material.count("b") / material.count("a") < 3.5
    assert "## Documents: Past interview / meeting transcript" in material


def test_terms_include_keywords_and_frequent_names():
    docs = [Doc("resume.txt", "Worked with PostgreSQL at Globex. Globex used PostgreSQL heavily. The end. The start.")]
    terms = transcription_terms(PROFILE, docs)
    assert terms[:2] == ["Kubernetes", "Acme"]
    assert "Globex" in terms and "PostgreSQL" in terms and "The" not in terms


def test_source_hash_changes_with_material_and_kind():
    a = source_hash(PROFILE, [Doc("a.txt", "one", "resume")])
    assert a == source_hash(PROFILE, [Doc("a.txt", "one", "resume")])
    assert a != source_hash(PROFILE, [Doc("a.txt", "two", "resume")])
    assert a != source_hash(PROFILE, [Doc("a.txt", "one", "prep")])


def test_extract_txt_and_docx():
    assert extract_text("notes.md", "﻿Hello   world\r\n\r\n\r\n\r\nBye".encode()) == "Hello world\n\nBye"

    import docx

    document = docx.Document()
    document.add_paragraph("Senior engineer at Globex")
    buffer = io.BytesIO()
    document.save(buffer)
    assert "Senior engineer at Globex" in extract_text("cv.docx", buffer.getvalue())
