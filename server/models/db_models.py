from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import time

class UserOut(BaseModel):
    username: str
    role: str

class ReportMeta(BaseModel):
    doc_id: str
    filename: str
    uploader: str
    uploaded_at: float
    num_chunks: int

class DiagnosisRecord(BaseModel):
    doc_id: str
    requester: str
    question: str
    answer: str
    source : Optional[List] = []
    timestamp: float = Field(default_factory=lambda: time.time())