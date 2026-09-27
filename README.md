<img width="1536" height="1024" alt="ZICO Travel Operations Assistant" src="https://github.com/user-attachments/assets/bfb73130-4ae3-4063-be5f-c0adc2076d78" />

# ZICO: Intelligent Travel Operations System

ZICO (**Speak. Plan. Go.**) is an enterprise-grade, multi-agent AI travel operations companion designed to manage the entire lifecycle of a journey. Unlike traditional chatbots that offer static text replies or invent flight schedules, ZICO maintains a verified, persistent state of the traveler’s itinerary, enabling proactive disruption management, deterministic constraint reasoning, grounded policy retrieval, real-time aviation intelligence, and automated recovery strategies with Human-in-the-Loop (HITL) authorization.

---

## System Architecture

```mermaid
graph TD
    User([Traveler / Web & Voice Client]) <--> NextJS[Next.js 15 Frontend UI]
    NextJS <--> API[FastAPI REST & Streaming Service]
    API <--> Graph[LangGraph Stateful Orchestrator]
    
    subgraph "LangGraph Agentic State Engine"
        Input[Input Normalizer] --> Router[Supervisor / Router Agent]
        Router -->|Flight Queries / Routes| FlightAgent[Flight Operations Agent]
        Router -->|Web Research / Visas| ResearchAgent[Tavily Research Agent]
        Router -->|Policy / Baggage / Rules| RAGWorker[Policy RAG Worker]
        Router -->|Delays / Cancellations| DisruptionWorker[Disruption Reasoning Worker]
        
        FlightAgent --> Validator[Deterministic Validator Agent]
        ResearchAgent --> Validator
        RAGWorker --> Validator
        DisruptionWorker --> Validator
        Validator --> ResponseAgent[Response Synthesis Agent]
        ResponseAgent --> Finish([Output TravelState / FINISH])
    end

    subgraph "Persistence, Caching & Data Subsystems"
        Postgres[(PostgreSQL: Itineraries & Immutable Audit Logs)]
        Redis[(Redis: Sliding Rate Limiter, Turn Memory & Semantic Cache)]
        Qdrant[(Qdrant Vector DB: Travel Policy Embeddings)]
        AviationStack[AviationStack: Live Flight Schedules & Statuses]
        Tavily[Tavily Search: Grounded Web Travel Intelligence]
        Whisper[OpenAI Whisper: Voice Transcription STT]
    end

    API --- Postgres
    API --- Redis
    RAGWorker --- Qdrant
    FlightAgent --- AviationStack
    ResearchAgent --- Tavily
    API --- Whisper
```

---

## Core Capabilities

### 1. Multi-Agent Orchestration (LangGraph Engine)
- **Supervisor Routing**: Natural-language intent classifier routing requests to specialized agents (`flight`, `research`, `policy`, `disruption`, `general_travel`).
- **Flight Operations Agent**: Integrates with AviationStack and offline `airportsdata` to query live schedules, delays, terminals, gates, and statuses without hallucinating flight numbers or times.
- **Tavily Research Agent**: Executes bounded, timeout-aware web search for destination guides, transit options, and travel advisories, preserving verifiable source citations.
- **Deterministic Validation**: Rigorously validates connection buffer deficits (minimum 90-minute international layovers), chronological consistency, and budget boundaries prior to answering.
- **Safe Response Agent**: Synthesizes verified data into structured, actionable travel plans while explicitly identifying external sources and stating when live data cannot be verified.

### 2. Contextual Policy RAG Subsystem
- **Vector Search with Qdrant**: Dense retrieval using OpenAI embeddings (`text-embedding-3-small`) over vetted travel regulations.
- **Pre-Seeded Travel Policies**:
  - **EU Regulation 261/2004**: Delay & cancellation compensation tiers (€250–€600) and duty-of-care provisions.
  - **US DOT 24-Hour Rule**: 100% full refund guarantees for flight cancellations within 24 hours of booking.
  - **IATA Baggage Standards**: Dimensions, carry-on limits (7kg–10kg), checked luggage standards (23kg), and excess fee regulations.
  - **Schengen Visa & Passport Validity**: 6-month validity rules and 90/180-day stay calculators.
  - **Travel Insurance Coverage**: Comprehensive trip cancellation, interruption reimbursement, and medical evacuation provisions.

### 3. Disruption Reasoning & Human-in-the-Loop (HITL) Controls
- **Ripple Effect Analyzer**: Detects cancellations or delays and computes downstream impacts on connecting flights, hotel check-ins, and ground transport.
- **Pending Action Proposals**: Generates structured `PendingAction` models with `requires_explicit_approval=True` for booking changes.
- **Immutable Audit Trail**: Logs every traveler decision (`APPROVE` / `REJECT`) with timestamps and context payloads in PostgreSQL `audit_logs`.

### 4. Memory, Semantic Caching & Rate Limiting
- **Redis Turn Memory**: Persists active conversation context with sliding 24-hour expiration and 30-turn limits.
- **Semantic Caching**: Caches safe travel policies and general research queries while enforcing a zero-cache policy on volatile real-time flight lookups (`flight` intent).
- **Sliding-Window Rate Limiting**: Protects high-cost endpoints (chat, search, voice transcription) with automatic in-memory fallback if Redis is unavailable.

### 5. Voice Subsystem
- **Speech-to-Text (STT)**: OpenAI Whisper endpoint (`/api/v1/voice/transcribe`) supporting `.wav`, `.mp3`, `.m4a`, and `.webm` formats with file-size caps and client-safe error normalization.
- **Text-to-Speech (TTS)**: Low-latency neural speech streaming with EdgeTTS and ElevenLabs.

---

## API Endpoints

### REST API (`/api/v1`)

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | `GET` | Health check for PostgreSQL, Redis, and Qdrant |
| `/api/v1/chat` | `POST` | Stateful travel inquiry execution via LangGraph |
| `/api/v1/trips` | `GET`, `POST` | Create and list active trip records |
| `/api/v1/trips/{trip_id}` | `GET`, `PUT` | Retrieve or update itinerary state and constraints |
| `/api/v1/actions/{action_id}/approve` | `POST` | Approve HITL recovery action and commit changes |
| `/api/v1/actions/{action_id}/reject` | `POST` | Reject HITL recovery action with audit log entry |
| `/api/v1/flights/search` | `POST` | Query live flight options via AviationStack |
| `/api/v1/rag/query` | `POST` | Semantic search across travel policy documents |
| `/api/v1/rag/index` | `POST` | Ingest and index custom policy documents into Qdrant |
| `/api/v1/voice/transcribe` | `POST` | Whisper audio transcription (STT) |
| `/api/v1/voice/synthesize` | `POST` | Neural text-to-speech audio streaming (TTS) |

### WebSocket API

| Endpoint | Protocol | Description |
|---|---|---|
| `/ws/stream/{trip_id}` | `WS` | Real-time bi-directional token streaming and node progress events |

---

## Getting Started

### Prerequisites
- Python 3.12+ or 3.14+
- Node.js 20+ and npm
- Docker and Docker Compose (for PostgreSQL, Redis, Qdrant)

### 1. Environment Configuration

Copy the example environment file and configure your API credentials:

```bash
cp .env.example .env
```

| Variable | Description | Default / Example |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API Key for Chat & Whisper | `sk-...` |
| `OPENAI_MODEL` | Chat completion model | `gpt-4o` |
| `AVIATIONSTACK_API_KEY` | AviationStack API Key for flight data | `live_...` |
| `TAVILY_API_KEY` | Tavily API Key for web research | `tvly-...` |
| `DATABASE_URL` | PostgreSQL async connection URI | `postgresql+asyncpg://zico:zicopass@localhost:5432/zicodb` |
| `REDIS_URL` | Redis connection URI | `redis://localhost:6379/0` |
| `QDRANT_HOST` | Qdrant vector database host | `localhost` |
| `QDRANT_PORT` | Qdrant vector database port | `6333` |
| `RATE_LIMIT_ENABLED` | Enable rate limiting middleware | `true` |
| `RATE_LIMIT_PER_MINUTE` | Max requests per minute per IP | `60` |

### 2. Start Supporting Infrastructure

```bash
docker-compose up -d
```

This launches PostgreSQL on `5432`, Redis on `6379`, and Qdrant on `6333`.

### 3. Initialize Database & Seed Knowledge

Run database schema migrations with Alembic:

```bash
cd backend
alembic upgrade head
```

Seed the baseline travel policies into Qdrant:

```bash
python -m app.rag.seed
```

### 4. Run the Backend API Server

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

- **Swagger Documentation**: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
- **ReDoc Documentation**: [http://localhost:8000/api/v1/redoc](http://localhost:8000/api/v1/redoc)

### 5. Run the Next.js Frontend

```bash
cd apps/web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Verification & Code Quality

Run tests, linting, formatting, and build checks from the repository root:

```bash
# 1. Run complete pytest test suite (511 tests)
python -m pytest

# 2. Verify Python compilation and import integrity
python -m compileall backend/app tests

# 3. Run Ruff code linter
python -m ruff check backend/app tests

# 4. Check code formatting
python -m ruff format --check backend/app tests

# 5. Build Next.js production bundle
cd apps/web && npm run build
```

---

## Security & Reliability Standards

- **Server-Side Privileged Operations**: All external services (OpenAI, AviationStack, Tavily, Qdrant, PostgreSQL, Redis) are accessed strictly from the backend. No secrets are exposed to the client.
- **Graceful Degradation**: If Redis or Qdrant is unavailable, the application falls back safely to in-memory caching/rate-limiting and direct knowledge workflows.
- **Anti-Hallucination Guarantees**: Flight statuses and numbers are never fabricated. If AviationStack or Tavily fails, ZICO explicitly communicates that live information could not be verified.
- **Prompt Injection Defense**: External search results and retrieved documents are treated as untrusted data and strictly isolated from system instructions.
- **Traceable Observability**: Every request is tagged with a unique `X-Request-ID` and structured log context.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
