from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from starlette.responses import Response as StarletteResponse
from ..auth.route import authenticate
from .vectorstore import load_vectorstore
import uuid
from typing import List
from ..config.db import reports_collection


def json_auth(user=Depends(authenticate)):
    # convert possible redirect/HTML responses into JSON 401
    if isinstance(user, StarletteResponse):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user

router=APIRouter(prefix="/reports",tags=["reports"])

@router.post("/upload")
async def upload_reports(user=Depends(json_auth),files:List[UploadFile]=File(...)):
    if user["role"] !="patient":
        raise HTTPException(status_code=403,detail="Only patients can upload reports for diagnosis")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    doc_id=str(uuid.uuid4())
    try:
        await load_vectorstore(files,uploaded=user["username"],doc_id=doc_id)
    except Exception as e:
        # Ensure JSON error response instead of server crash/non-JSON
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    return JSONResponse({"message":"Uploaded and indexed","doc_id":doc_id})