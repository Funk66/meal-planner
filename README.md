# AI Meal Planner

A local-first, privacy-focused meal planner that learns your eating habits and improves over time.

## Motivaton

Search online for "AI meal planner" and you'll find countless shady websites offering exactly that: pay a subscription and an LLM creates a personalized plan for you. Magic! ✨

While some of these may do a decent job at that, you should be mindful about sharing sensitive information with unknown third parties - your dietary preferences, restrictions, goals, etc.

The good news is that you don't need a massive SOTA LLM for a job like this. If you can spare 8GB of memory on your personal laptop, you can do just the same, privately, and without having to shell out for yet another subscription.

## How It Works

1. **Populate the database first.** The ingestion pipeline downloads the Food.com dataset, parses ingredient strings into structured objects (quantity/unit/notes), embeds ingredient names, and writes everything to Postgres.
2. **Ask questions with the CLI.** The query is embedded and used to retrieve similar recipes, which are then adapted by a local LLM and sanity-checked for unsafe instructions.

## Tech Stack

- **Python 3.14** (managed by `uv`)
- **Orchestration:** LangGraph
- **LLM/Embeddings:** Ollama (local)
- **Database:** PostgreSQL + pgvector
- **Data Source:** Food.com Recipes (via Kaggle)

## Quick Start

### 1. Prerequisites

- Install [uv](https://github.com/astral-sh/uv)
- Install [Docker](https://www.docker.com/)
- Install [Ollama](https://ollama.com/) and pull models:
  ```bash
  ollama pull qwen2.5:7b
  ollama pull nomic-embed-text
  ```

### 2. Configure Environment

Copy `.env.example` to `.env` and update as needed:

```bash
POSTGRES_CONNECTION_STRING=postgresql://meal:meal@localhost:5432/mealplanner
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Start Postgres

```bash
docker-compose up -d
```

### 4. Install Dependencies

```bash
uv sync
```

## Populate The Database

Recipe suggestions are not hallucinated by the LLM, but based on real recipes from the Food.com dataset.
Run the following command to get the database ready:

```bash
uv run meal-planner-ingest
```

The ingestion script may take a long time to complete, depending on your local resources.
You can stop the process and resume later where it left off.

## Use The CLI

Ask a question directly:

```bash
uv run meal-planner "I have leftover chicken, what can I make?"
```
