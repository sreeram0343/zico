# ZICO: How to Run the Project

This guide provides complete, step-by-step instructions to set up, configure, run, and test the **ZICO Intelligent Travel Operations Assistant** locally.

---

## 📋 Prerequisites

Before starting, ensure your machine has the following tools installed:

| Tool | Recommended Version | Check Command |
|---|---|---|
| **Python** | 3.12 or newer | `python --version` |
| **Node.js** | 20.x or newer (18+ supported) | `node --version` |
| **npm** | 10.x or newer | `npm --version` |
| **Docker & Docker Compose** | Latest Desktop version | `docker --version` and `docker compose version` |
| **Git** | Latest | `git --version` |

---

## ⚡ Quick Start (TL;DR)

If you already have prerequisites installed and just want to get up and running:

### Terminal 1: Backend
```bash
# 1. Start database, cache & vector storage (optional, fallback available)
docker compose up -d

# 2. Setup environment file
cp .env.example .env

# 3. Install Python dependencies
pip install -r backend/requirements.txt

# 4. Run database migrations
cd backend && alembic upgrade head

# 5. Start FastAPI Backend (Port 8000)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2: Frontend
```bash
# 1. Navigate to frontend directory
cd apps/web

# 2. Install dependencies
npm install

# 3. Start Next.js Development Server (Port 3000)
npm run dev
```

Visit **http://localhost:3000** in your browser.

---

## 🛠️ Step-by-Step Setup Guide

### Step 1: Environment Variables Configuration

Copy the sample environment file to `.env` in the repository root:

```bash
cp .env.example .env
```

Open `.env` in your text editor and adjust keys as needed:

```ini
# Application Mode & Logging
APP_ENV=development
LOG_LEVEL=INFO

# OpenAI API Key (Required for LLM and Whisper STT)
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL=gpt-4o-mini

# Aviation Flight Data (Optional - Graceful degradation if missing)
AVIATIONSTACK_API_KEY=your_aviationstack_key
DEFAULT_ORIGIN_IATA=BOM

# Travel Research Search Provider (Optional)
TAVILY_API_KEY=your_tavily_api_key

# PostgreSQL Persistent Database
POSTGRES_USER=user
POSTGRES_PASSWORD=pass
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=zico
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/zico

# Redis Cache, Session Memory & Rate Limiting
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_URL=redis://localhost:6379/0

# Qdrant Vector Database (Travel Policy Knowledge Base)
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=travel_policies

# Frontend URLs
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

> **Note:** If external credentials (`AVIATIONSTACK_API_KEY`, `TAVILY_API_KEY`, `REDIS_URL`, or `QDRANT_URL`) are omitted, ZICO gracefully uses built-in offline airport databases, safe deterministic fallback responses, and in-memory caches.

---

### Step 2: Start Supporting Infrastructure with Docker

Launch PostgreSQL, Redis, and Qdrant using the project's root `docker-compose.yml`:

```bash
docker compose up -d
```

Verify that the containers are healthy and running:

```bash
docker compose ps
```

You should see:
- `zico_postgres` on port `5432`
- `zico_redis` on port `6379`
- `zico_qdrant` on ports `6333` and `6334`

---

### Step 3: Set Up and Run the Backend Server

1. **(Recommended) Create and activate a Python virtual environment:**

   On **Windows (PowerShell)**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   On **macOS / Linux**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Apply Database Migrations:**
   ```bash
   cd backend
   alembic upgrade head
   cd ..
   ```

4. **(Optional) Seed Knowledge Base into Qdrant:**
   To populate the vector database with baseline travel regulations (EU261, US DOT 24-hr rule, baggage allowances):
   ```bash
   python -m backend.app.rag.seed
   ```

5. **Start the FastAPI Backend Service:**
   ```bash
   cd backend
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

   - **Backend Health Check**: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
   - **Interactive API Documentation (Swagger)**: [http://127.0.0.1:8000/api/v1/docs](http://127.0.0.1:8000/api/v1/docs)
   - **Alternative ReDoc UI**: [http://127.0.0.1:8000/api/v1/redoc](http://127.0.0.1:8000/api/v1/redoc)

---

### Step 4: Set Up and Run the Frontend UI

In a new terminal window:

1. **Navigate to the frontend web application:**
   ```bash
   cd apps/web
   ```

2. **Install Node dependencies:**
   ```bash
   npm install
   ```

3. **Start the Next.js development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser:**
   Navigate to **[http://localhost:3000](http://localhost:3000)**.

You will see the **ZICO Intelligent Travel Operations Assistant** yellow-and-white dashboard:
- **Left Sidebar**: Aviation logo, active navigation, settings, and travel illustration banner.
- **Header**: Visual theme toggle and user profile account menu.
- **Hero Banner**: Panoramic travel banner with airplane over Santorini coastal scenery.
- **Quick Action Pills**: Find Flights, Plan Itinerary, Explore Destinations, Travel Research.
- **Interactive Chat Workspace**: Message bubbles, airline flight result cards with "View Details" modals, quick follow-up action pills, and voice input composer.
- **Right Sidebar**: Travel Assistant info, popular travel questions, and recent conversations.

---

## 🧪 Verification & Automated Tests

To verify complete application health, test suites, and code standards:

### 1. Run Backend Pytest Suite (527 Tests)
From the repository root:
```bash
python -m pytest
```

### 2. Verify Python Syntax & Compilation
```bash
python -m compileall backend/app tests
```

### 3. Run Code Quality & Formatting Checks (Ruff)
```bash
# Check code style and imports
python -m ruff check backend/app tests

# Verify formatting
python -m ruff format --check backend/app tests
```

### 4. Build Production Frontend Bundle
```bash
cd apps/web
npm run build
```

---

## 🔌 API & Communication Architecture

| Port | Service | Description |
|---|---|---|
| `3000` | **Next.js Web Frontend** | Responsive yellow-and-white travel operations UI |
| `8000` | **FastAPI Backend Server** | REST endpoints (`/api/v1/*`) and WebSocket streaming (`/ws/stream/{trip_id}`) |
| `5432` | **PostgreSQL Database** | Persistent trip itineraries, constraints, and audit logs |
| `6379` | **Redis Cache & Memory** | Sliding-window rate limiting, session cache, and multi-turn state |
| `6333` | **Qdrant Vector DB** | Semantic embeddings for travel policy retrieval (RAG) |

---

## ❓ Troubleshooting & FAQs

### Q1: Can I run ZICO without Docker?
**Yes.** If Docker is not running on your machine, ZICO automatically falls back to:
- In-memory rate limiting and memory caching.
- Direct rule-based policy matching if Qdrant is unavailable.
- SQLite / local mock storage if PostgreSQL is unreachable.

### Q2: Port 8000 or 3000 is already in use.
- **Backend:** Change port via `python -m uvicorn app.main:app --port 8080` and update `NEXT_PUBLIC_API_URL=http://localhost:8080` in `.env.local`.
- **Frontend:** Run `npm run dev -- -p 3001` to start Next.js on port 3001.

### Q3: How do I test voice transcription?
Ensure your browser allows microphone access on `http://localhost:3000`. When you click the microphone button, speaking triggers recording, audio chunks are sent to `/api/v1/voice/transcribe`, and the transcribed text is automatically populated into the chat input.

### Q4: WebSocket connection shows disconnected.
Ensure the backend server is running on `http://127.0.0.1:8000`. The frontend automatically tries connecting to `ws://localhost:8000/ws/stream/{trip_id}` with an automatic HTTP fallback to `POST /api/v1/chat` if WebSocket is unavailable.
