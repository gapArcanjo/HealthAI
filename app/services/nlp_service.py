from app.models.schemas import EntityExtraction, IntentType, NLPAnalysis, UrgencyLevel


EMERGENCY_KEYWORDS = (
    "dor no peito",
    "falta de ar",
    "desmaio",
    "convulsao",
    "sangramento intenso",
)


class RuleBasedNLPService:
    def analyze(self, text: str) -> NLPAnalysis:
        msg = text.lower().strip()
        symptoms = [k for k in EMERGENCY_KEYWORDS if k in msg]
        is_emergency = len(symptoms) > 0

        if "agendar" in msg or "consulta" in msg:
            intent = IntentType.SCHEDULING
        elif "cancel" in msg:
            intent = IntentType.CANCELLATION
        elif "exame" in msg:
            intent = IntentType.EXAM_INQUIRY
        elif symptoms or "dor" in msg or "sintoma" in msg:
            intent = IntentType.SYMPTOMS
        else:
            intent = IntentType.GENERAL

        urgency = UrgencyLevel.CRITICAL if is_emergency else UrgencyLevel.MEDIUM
        return NLPAnalysis(
            intent=intent,
            intent_confidence=0.9,
            urgency=urgency,
            urgency_confidence=0.9 if is_emergency else 0.7,
            entities=EntityExtraction(symptoms=symptoms),
            is_emergency=is_emergency,
            recommended_specialty="pronto_socorro" if is_emergency else None,
        )


nlp_service = RuleBasedNLPService()
