from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from .models import SignupRequest
from .hash_util import hash_password, verify_password
from ..config.db import users_collection

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    user = users_collection.find_one({"username": credentials.username})
    if not user or not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": user["username"], "role": user["role"]}

@router.post("/signup")
def signup(request: SignupRequest):
    if users_collection.find_one({"username": request.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_password = hash_password(request.password)
    users_collection.insert_one({"username": request.username, "password": hashed_password, "role": request.role})
    return {"message": "User created successfully"}

@router.get("/login")
def login(user: dict = Depends(authenticate)):
    return {"message": user['username'], "role": user["role"]}
