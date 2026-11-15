"""SuccessGuideAI: a course and career assistant that answers from the student's own materials."""

from .store import CourseLibrary
from .assistant import Assistant, Answer

__all__ = ["CourseLibrary", "Assistant", "Answer"]
