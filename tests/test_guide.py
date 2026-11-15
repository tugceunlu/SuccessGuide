"""Tests for the library and the assistant. The chat model is a fake, so no API key is needed."""

from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from guide import Assistant, CourseLibrary
from guide.assistant import format_context

SAMPLES = Path(__file__).parent.parent / "sample_materials"


def make_library() -> CourseLibrary:
    lib = CourseLibrary(path=None)
    lib.clear()
    lib.add_folder(SAMPLES)
    return lib


def test_upload_twice_does_not_duplicate():
    lib = make_library()
    n = lib.count()
    lib.add_folder(SAMPLES)
    assert lib.count() == n and len(lib.sources()) == 3


def test_search_returns_the_matching_course():
    lib = make_library()
    assert lib.search("How do I write good learning objectives?")[0].source == "instructional_design.txt"


def test_assistant_sends_context_and_keeps_history():
    lib = make_library()
    llm = FakeListChatModel(responses=["First answer [1]", "Second answer"])
    bot = Assistant(lib, llm)
    a1 = bot.ask("How can I succeed in the data analysis course?", interests="statistics")
    assert a1.text == "First answer [1]" and a1.sources
    bot.ask("And after that?")
    assert len(bot.history) == 4


def test_context_mentions_when_no_materials():
    assert "no course materials" in format_context([])
