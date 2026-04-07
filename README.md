# 🧠 HealthAI Assistant

API de assistente de saúde desenvolvida com **FastAPI**, estruturada com princípios de **Clean Architecture**, incorporando um **Agente de IA** para processamento inteligente de linguagem natural e tomada de decisão.

---

## 📌 Visão Geral

O HealthAI Assistant implementa um **Agente de IA orientado a tarefas**, capaz de:

* Interpretar mensagens de usuários
* Classificar intenções (ex: emergência médica)
* Tomar decisões com base em contexto
* Gerar respostas com modelos de linguagem (LLMs)

A arquitetura foi projetada para:

* Separação clara de responsabilidades
* Escalabilidade
* Baixo acoplamento
* Facilidade de evolução para sistemas baseados em agentes

---

## 🤖 Agente de IA

O sistema implementa uma **arquitetura baseada em agentes**, onde um componente central coordena o fluxo de decisão.

### 🔹 Responsabilidades do agente

* Processamento de entrada do usuário
* Classificação semântica (NLP)
* Tomada de decisão baseada em regras + LLM
* Geração de resposta contextual

---

### 🔹 Arquitetura do agente

```text
User Input
   ↓
Orchestrator (Agent)
   ↓
NLP Service → Intent Classification
   ↓
LLM Service → Response Generation
   ↓
Final Response
```

---

### 🔹 Tipo de agente

O HealthAI utiliza um modelo híbrido:

* **Rule-based layer** → decisões críticas (ex: emergência)
* **LLM-based reasoning** → geração de linguagem natural
* **Orchestrator** → controlador central do agente

---

## 🏗️ Estrutura do Projeto

```bash
health-ai-assistant/
├── app/                  # Código principal da aplicação (API + agente)
├── frontend/             # Interface (opcional)
├── ml/                   # Pipeline de Machine Learning
├── tests/                # Testes automatizados
├── Dockerfile            # Build da aplicação
├── docker-compose.yml    # Orquestração de containers
├── requirements.txt      # Dependências
└── README.md
```

---

## 🧩 Arquitetura (Clean Architecture)

Dentro da pasta `app/`:

### 🔹 Domain

* Modelos e regras de negócio

### 🔹 Application

* Orquestração do agente (core logic)

### 🔹 Infrastructure

* Serviços externos (NLP, LLM)

### 🔹 Interface

* Endpoints FastAPI

### 🔹 Cross-cutting

* Configuração e logging

---

## 🔄 Fluxo da Aplicação

```text
Client → FastAPI → AI Agent (Orchestrator) → NLP/LLM → Response
```

---

## 🤖 Módulo de Machine Learning

```bash
ml/
├── training.py
├── cross_validation.py
├── inference.py
```

Responsabilidades:

* Treinamento de modelos
* Validação
* Inferência offline

---

## 📡 Endpoints

* `GET /health`
* `GET /`
* `POST /api/v1/chat/message`

### Exemplo:

```json
{
  "session_id": "sess-001",
  "message": "Estou com dor no peito"
}
```

---

## ⚙️ Variáveis de Ambiente

* `OPENAI_API_KEY`
* `LLM_MODEL`
* `LLM_TEMPERATURE`
* `EMERGENCY_PHONE`

---

## ▶️ Executando Localmente

```bash
python -m venv .venv
```

Ativação:

* Linux/Mac:

```bash
source .venv/bin/activate
```

* Windows:

```bash
.venv\Scripts\activate
```

Instalação:

```bash
pip install -r requirements.txt
```

Execução:

```bash
uvicorn app.main:app --reload
```

Docs:
http://localhost:8000/docs

---

## 🐳 Docker

```bash
docker compose up --build -d
```

---

## 🧪 Testes

```bash
pytest
```

---

## ✅ Boas Práticas

* Clean Architecture
* Inversão de dependência
* Arquitetura baseada em agentes
* Código modular e testável
* Baixo acoplamento entre camadas
* Preparado para evolução com IA
