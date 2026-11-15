"""Answering a student's questions from their course materials (retrieval-augmented generation)."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .store import CourseLibrary, Passage

SYSTEM_PROMPT = """You are SuccessGuide, a supportive study and career advisor for university students.

Rules:
- Base course-specific advice on the course excerpts below. Refer to them as [1], [2], ...
- If the excerpts do not contain the answer, say so plainly and give general advice, clearly marked as general.
- Take the student's interests into account when suggesting study strategies or careers.
- Be concrete and brief: a short answer, then at most five bullet points.
- Encourage the student to make their own decisions; do not decide for them.

Student's interests: {interests}

Course excerpts:
{context}"""


@dataclass
class Answer:
    text: str
    sources: list[Passage] = field(default_factory=list)


def format_context(passages: list[Passage]) -> str:
    if not passages:
        return "(no course materials uploaded)"
    return "\n\n".join(f"[{i}] ({p.source}) {p.text}" for i, p in enumerate(passages, start=1))


class Assistant:
    def __init__(self, library: CourseLibrary, llm: BaseChatModel, k: int = 4, history_turns: int = 6):
        self.library = library
        self.llm = llm
        self.k = k
        self.history_turns = history_turns
        self.history: list[HumanMessage | AIMessage] = []

    def ask(self, question: str, interests: str = "") -> Answer:
        passages = self.library.search(f"{interests} {question}".strip(), k=self.k)
        system = SystemMessage(
            SYSTEM_PROMPT.format(interests=interests or "not given", context=format_context(passages))
        )
        recent = self.history[-2 * self.history_turns :]
        reply = self.llm.invoke([system, *recent, HumanMessage(question)])
        self.history += [HumanMessage(question), AIMessage(reply.content)]
        return Answer(text=reply.content, sources=passages)

    def reset(self) -> None:
        self.history.clear()
