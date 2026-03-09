from fastapi import APIRouter

router = APIRouter()


@router.get("/ping")
async def ping_auth():
    return {"status": "ok", "service": "auth"}
