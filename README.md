# ZICO: Intelligent Travel Operations System

ZICO (Speak. Plan. Go.) is a voice-first, context-aware AI travel companion designed to manage the entire lifecycle of a journey. Unlike traditional chatbots that provide static recommendations, ZICO maintains a persistent state of the traveler’s itinerary, enabling proactive disruption management, real-time constraint reasoning, and automated recovery strategies.

## Core Philosophy

ZICO treats travel as a state-based system. By utilizing a central source of truth for all bookings, preferences, and constraints, the system ensures that every decision—whether booking a flight or adjusting an itinerary due to a delay—is validated against the user's complete context.

## Key Capabilities

- **Persistent State Management**: Maintains a synchronized representation of flights, accommodations, activities, and budget across the entire trip.
- **Disruption Reasoning**: Automatically detects external changes (flight cancellations, weather delays) and calculates the ripple effect on downstream connections and bookings.
- **Agentic Orchestration**: Uses a supervisor-worker architecture to delegate tasks to specialized tools for flight searches, policy retrieval, and route optimization.
- **Human-in-the-Loop Controls**: Ensures high-impact actions—such as payments, cancellations, or rebookings—require explicit user verification before execution.
- **Contextual RAG**: Retrieves real-time policy information, visa requirements, and insurance documentation to provide grounded, reliable guidance.

## Architecture

ZICO is built as a modular AI engineering system:

- **Orchestration**: LangGraph (Stateful Agentic Workflows)
- **Frontend**: Next.js 15 (App Router, Tailwind CSS, TypeScript)
- **Backend**: FastAPI (Python 3.12+, Asynchronous Services)
- **Data Persistence**: PostgreSQL (Structured Data) & Redis (Session State)
- **Semantic Search**: Qdrant (Vector Database for Policy RAG)
- **Voice Pipeline**: OpenAI Whisper (STT) and ElevenLabs/EdgeTTS (TTS)

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.12+
- Node.js 20+
- API Keys (OpenAI, Amadeus, Qdrant Cloud)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sreeram0343/zico.git
   cd zico
   ```

2. Initialize environment variables:
   ```bash
   cp .env.example .env # Configure required API keys in the .env file
   ```

3. Launch infrastructure services:
   ```bash
   docker-compose up -d
   ```

4. Install dependencies and run development servers:
   ```bash
   # Backend
   cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload

   # Frontend
   cd ../apps/web && npm install && npm run dev
   ```

## Development Workflow

This project utilizes a modular contribution strategy. Contributors are expected to follow atomic commit practices:

- `feat`: New functionality or integration.
- `fix`: Bug resolution or state correction.
- `test`: Unit or integration test coverage.
- `chore`: Maintenance and dependency updates.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration inquiries, please open an issue in this repository.