from fastapi import APIRouter, Depends, Form, HTTPException
from ..auth.route import authenticate
from .query import diagnosis_report
from ..config.db import reports_collection, diagnosis_collection
import time
from fastapi.responses import JSONResponse
from starlette.responses import Response as StarletteResponse

def json_auth(user=Depends(authenticate)):
    if isinstance(user, StarletteResponse):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user

router=APIRouter(prefix="/diagnosis",tags=["diagnosis"])

@router.post("/from_report")
async def diagnos(user=Depends(json_auth),doc_id:str=Form(...),question:str=Form(default="Please provide a diagnosis based on my report")):
    report=reports_collection.find_one({"doc_id":doc_id})
    if not report:
        raise HTTPException(status_code=404,detail="Report not found")
    
    # patient can only access
    if user["role"] == "patient" and report["uploader"] != user["username"]:
        raise HTTPException(status_code=406,detail="You cannot access another user's report")
    
    # if user is a patient and want diagnosis from his own report
    if user["role"]=="patient":
        try:
            res=await diagnosis_report(user["username"],doc_id,question)
        except Exception as e:
            # Ensure JSON error response instead of non-JSON 500
            raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(e)}")

        # persist the diagnosis report (best-effort)
        try:
            diagnosis_collection.insert_one({
                "doc_id": doc_id,
                "requester": user["username"],
                "question": question,
                "answer": res.get("diagnosis"),
                "sources": res.get("sources", []),
                "timestamp": time.time()
            })
        except Exception:
            # Do not fail the response if persistence fails
            res["warning"] = "Diagnosis saved partially; persistence failed."
        return JSONResponse(res)
    
    # if the user is a doctor or other, then they can't ask for diagnosis
    if user["role"] in ("doctor","admin"):
        raise HTTPException(status_code=407,detail="Doctors cannot access for diagnosis with this endpoint")
    
    raise HTTPException(status_code=408,detail="Unauthorized action")


@router.get("/by_patient_name")
async def get_patient_diagnosis(patient_name: str, user=Depends(json_auth)):
    # Only doctors can view a patient's diagnosis
    if user["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can access this endpoint")
        
    diagnosis_records = list(diagnosis_collection.find({"requester": patient_name}))
    if not diagnosis_records:
        raise HTTPException(status_code=404, detail="No diagnosis found for this patient")
        
    for record in diagnosis_records:
        record["_id"] = str(record["_id"])
    return JSONResponse(diagnosis_records)