import pytest

from app.detector import is_echo, is_question


@pytest.mark.parametrize(
    "text",
    [
        "Can you tell me about yourself?",
        "So what made you apply for this role",
        "Tell me about a time you disagreed with a manager.",
        "Walk me through your last project.",
        "Okay great. How would you scale this service",
        "Describe a situation where you had to learn fast.",
        "Great, thanks. Why do you want to leave your current job",
        "I'd love to hear how you handled the outage.",
        "Do you have any questions for us",
    ],
)
def test_detects_questions(text):
    assert is_question(text)


@pytest.mark.parametrize(
    "text",
    [
        "Thanks, that makes sense.",
        "We are a team of twelve engineers based in Berlin.",
        "Let me explain our process a bit first.",
        "Okay.",
        "Why?",  # too short to be worth an automatic answer
        "Our market share grew last year.",
    ],
)
def test_ignores_statements(text):
    assert not is_question(text)


def test_echo_of_their_speech_is_detected():
    theirs = ["So tell me about the biggest project you shipped last year and what you learned."]
    assert is_echo("tell me about the biggest project you shipped last year", theirs)
    assert is_echo("So tell me about the biggest project you shipped last year and what you learned", theirs)


def test_my_own_words_are_not_echo():
    theirs = ["So tell me about the biggest project you shipped last year."]
    assert not is_echo("Sure, the biggest one was a payments platform rewrite at Acme.", theirs)
    assert not is_echo("Yes.", ["Yes, exactly, and then we moved on to the next topic."])
    assert not is_echo("anything", [])
