# Meal Planner: Local RAG Recipe Planner

A local-first, privacy-focused meal planner that uses LLMs suggest recipes based on user input.

## 🛠 Tech Stack

- **Python 3.14** (Managed by `uv`)
- **Orchestration:** LangGraph
- **LLM/Embeddings:** Ollama (local)
- **Database:** PostgreSQL + pgvector
- **Data Source:** Food.com Recipes (via Kaggle)

## 🚀 Quick Start

### 1. Prerequisites

- Install [uv](https://github.com/astral-sh/uv)
- Install [Docker](https://www.docker.com/)
- Install [Ollama](https://ollama.com/) and pull models:
  ```bash
  ollama pull llama3
  ollama pull nomic-embed-text
  ```
