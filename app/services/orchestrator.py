import asyncio

from app.models.schemas import IntentType, MessageRequest, MessageResponse, NLPAnalysis, UrgencyLevel
from app.services.nlp_service import nlp_service
from app.utils.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

EMERGENCY_RESPONSE = (
    "ATENCAO: possivel emergencia detectada. Ligue agora para 192 (SAMU) "
    "ou procure o pronto-socorro mais proximo."
)


class HealthAIOrchestrator:
    """Coordinates the NLP classification and response strategy."""

    async def process_message(self, request: MessageRequest) -> MessageResponse:
        analysis = await self._analyze(request.message)

        if analysis.is_emergency:
            return self._emergency_response(request, analysis)

        return MessageResponse(
            session_id=request.session_id,
            message=self._fallback_response(analysis),
            nlp_analysis=analysis,
            suggested_actions=self._suggested_actions(analysis),
            emergency_protocol=False,
        )

    async def _analyze(self, text: str) -> NLPAnalysis:
        loop = asyncio.get_running_loop()
        analysis = await loop.run_in_executor(None, nlp_service.analyze, text)
        logger.info(
            "classifier intent=%s urgency=%s emergency=%s",
            analysis.intent,
            analysis.urgency,
            analysis.is_emergency,
        )
        return analysis

    def _emergency_response(self, request: MessageRequest, analysis: NLPAnalysis) -> MessageResponse:
        return MessageResponse(
            session_id=request.session_id,
            message=EMERGENCY_RESPONSE,
            nlp_analysis=analysis,
            suggested_actions=["Ligar 192 (SAMU)", "Ir ao pronto-socorro"],
            emergency_protocol=True,
            emergency_message=f"Protocolo de emergencia ativado. Contato: {settings.emergency_phone}",
        )

    def _suggested_actions(self, analysis: NLPAnalysis) -> list[str]:
        actions: list[str] = []
        if analysis.intent == IntentType.SCHEDULING:
            actions.append("Agendar consulta")
        if analysis.intent == IntentType.CANCELLATION:
            actions.append("Ver agendamentos")
        if analysis.urgency in (UrgencyLevel.HIGH, UrgencyLevel.CRITICAL):
            actions.append("Ir ao pronto-socorro")
        if not actions:
            actions.append("Falar com atendente")
        return actions

    def _fallback_response(self, analysis: NLPAnalysis) -> str:
        if analysis.intent == IntentType.SCHEDULING:
            return "Posso ajudar no agendamento. Informe nome completo e data desejada."
        if analysis.intent == IntentType.SYMPTOMS:
            return "Descreva seus sintomas e ha quanto tempo eles comecaram."
        if analysis.intent == IntentType.EXAM_INQUIRY:
            return "Informe o numero do pedido de exame ou CPF para consulta."
        return "Sou o HealthAI Assistant. Posso ajudar com sintomas e agendamentos."


orchestrator = HealthAIOrchestrator()
