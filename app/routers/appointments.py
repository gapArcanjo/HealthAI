from fastapi import APIRouter

router = APIRouter()


@router.get("/ping")
async def ping_appointments():
    return {"status": "ok", "service": "appointments"}
