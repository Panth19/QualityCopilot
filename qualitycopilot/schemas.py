from __future__ import annotations
from pydantic import BaseModel, Field

class Hypothesis(BaseModel):
    claim: str = Field(min_length=8)
    citations: list[str] = Field(min_length=1)

class Analysis(BaseModel):
    summary: str
    hypotheses: list[Hypothesis]

class SupportCheck(BaseModel):
    supported: bool
    reason: str
