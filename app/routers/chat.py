from fastapi import APIRouter

from app.models.schemas import MessageRequest, MessageResponse
from app.services.orchestrator import orchestrator

router = APIRouter()


@router.post("/message", response_model=MessageResponse)
async def send_message(request: MessageRequest) -> MessageResponse:
    return await orchestrator.process_message(request)
