# HealthAI Assistant

API de assistente de saude com FastAPI, organizada com principios de Clean Architecture.

## Visao Geral

O projeto separa regras de negocio, casos de uso, adaptadores de infraestrutura e camada HTTP para facilitar manutencao, testes e evolucao.

## Arquitetura (Clean)

- Domain (`app/models`): entidades, enums e contratos de dados (Pydantic).
- Application (`app/agents`): orquestracao de casos de uso.
- Infrastructure (`app/services`): implementacoes concretas (NLP e LLM/OpenAI).
- Interface/Presentation (`app/routers`, `app/main.py`): endpoints e composicao da API.
- Cross-cutting (`app/utils`): configuracao e logging.

## Estrutura do Projeto

```text
health-ai-assistant/
+-- app/
¦   +-- main.py
¦   +-- agents/
¦   ¦   +-- orchestrator.py
¦   +-- models/
¦   ¦   +-- schemas.py
¦   +-- routers/
¦   ¦   +-- chat.py
¦   +-- services/
¦   ¦   +-- ports.py
¦   ¦   +-- nlp_service.py
¦   ¦   +-- llm_service.py
¦   +-- utils/
¦       +-- config.py
¦       +-- logger.py
+-- ml/
¦   +-- training.py
¦   +-- cross_validation.py
¦   +-- inference.py
+-- infra/
¦   +-- dockerfile
¦   +-- docker-compose.yml
¦   +-- terraform/
+-- docs/
¦   +-- demo/
¦       +-- healthai-demo.html
+-- tests/
+-- pyrightconfig.json
+-- README.md
```

## Endpoints

- `GET /health`: status da API.
- `GET /`: informacoes basicas.
- `POST /api/v1/chat/message`: fluxo de mensagem (classificacao + resposta).

Exemplo de request:

```json
{
  "session_id": "sess-001",
  "message": "Estou com dor no peito"
}
```

## Variaveis de Ambiente

Configuradas em `app/utils/config.py`:

- `ENV` (default: `development`)
- `ALLOWED_ORIGINS` (default: `*`)
- `TRUSTED_HOSTS` (default: `*`)
- `OPENAI_API_KEY` (opcional)
- `LLM_MODEL` (default: `gpt-4o-mini`)
- `LLM_MAX_TOKENS` (default: `300`)
- `LLM_TEMPERATURE` (default: `0.4`)
- `EMERGENCY_PHONE` (default: `192`)

## Executando Localmente

1. Crie e ative o ambiente virtual.
2. Instale dependencias.
3. Inicie a API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Acesse: [http://localhost:8000/docs](http://localhost:8000/docs)

## Docker

Com o compose dentro de `infra/`:

```bash
cd infra
docker compose up -d
```

## Qualidade de Codigo

- Dependencia invertida via `ports.py`.
- Orquestracao desacoplada das implementacoes concretas.
- Modelos centralizados e tipados.
- Organizacao por responsabilidade para facilitar testes.
