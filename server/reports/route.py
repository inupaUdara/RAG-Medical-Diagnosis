from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List
import uuid
from ..auth.route import authenticate
from ..config.db import reports_collection
from .vectorstore import load_vectorstore

router = APIRouter(prefix="/reports", tags=["reports"])

@router.post("/upload")
async def upload_reports(
    files: List[UploadFile] = File(...),
    user=Depends(authenticate)
):
    if user["role"] != "patient":
        raise HTTPException(status_code=403, detail="Not authorized to upload reports")

    doc_id = str(uuid.uuid4())
    await load_vectorstore(files, uploaded=user["username"], doc_id=doc_id)

    return {"message": "Reports uploaded successfully", "doc_id": doc_id}
