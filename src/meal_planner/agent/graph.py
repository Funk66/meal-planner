from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, StateGraph
from pgvector.psycopg import Vector, register_vector
from psycopg import connect

DEFAULT_LLM_MODEL = "qwen2.5:7b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_TOP_K = 5

CRITIC_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "safe": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["safe", "issues"],
    "additionalProperties": False,
}


class GraphState(TypedDict, total=False):
    query: str
    recipes: list[dict]
    response: str
    critique: dict


@dataclass
class Settings:
    postgres_connection_string: str
    ollama_base_url: str
    embed_model: str
    llm_model: str


def load_settings() -> Settings:
    load_dotenv()
    connection = os.getenv("POSTGRES_CONNECTION_STRING")
    if not connection:
        raise RuntimeError("POSTGRES_CONNECTION_STRING is not set in the environment")
    return Settings(
        postgres_connection_string=connection,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        embed_model=os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        llm_model=os.getenv("OLLAMA_LLM_MODEL", DEFAULT_LLM_MODEL),
    )


def _format_quantity(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


def _format_ingredient_lines(ingredients: list[dict]) -> list[str]:
    lines: list[str] = []
    for item in ingredients:
        if not isinstance(item, dict):
            continue
        search_term = str(item.get("search_term", "")).strip()
        if not search_term:
            continue
        quantity = item.get("quantity")
        unit = str(item.get("unit", "")).strip()
        notes = str(item.get("notes", "")).strip()
        qty_part = ""
        if isinstance(quantity, (int, float)):
            qty_part = _format_quantity(float(quantity))
        parts = [part for part in [qty_part, unit, search_term] if part]
        line = " ".join(parts)
        if notes:
            line = f"{line} ({notes})"
        lines.append(line)
    return lines


def _retrieve_factory(settings: Settings, embedder: OllamaEmbeddings, top_k: int):
    def retrieve(state: GraphState) -> GraphState:
        query = state["query"]
        embedding = Vector(embedder.embed_query(query))
        with connect(settings.postgres_connection_string) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, ingredients, instructions, tags,
                           (embedding <=> %s) AS distance
                    FROM recipes
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (embedding, embedding, top_k),
                )
                rows = cur.fetchall()
        recipes = [
            {
                "id": row[0],
                "name": row[1],
                "ingredients": row[2],
                "instructions": row[3],
                "tags": row[4],
                "distance": row[5],
            }
            for row in rows
        ]
        return {"recipes": recipes}

    return retrieve


def _generator_factory(settings: Settings, llm: ChatOllama):
    def generate(state: GraphState) -> GraphState:
        query = state["query"]
        recipes = state.get("recipes", [])
        if not recipes:
            return {
                "response": f"No recipes found for '{query}'. Try a different query.",
            }

        blocks: list[str] = []
        for recipe in recipes[:3]:
            ingredient_lines = _format_ingredient_lines(recipe.get("ingredients", []))
            ingredients_text = "\n".join(
                f"- {line}" for line in ingredient_lines
            )
            steps = recipe.get("instructions", [])
            steps_text = "\n".join(
                f"{idx + 1}. {step}" for idx, step in enumerate(steps)
            )
            tags = recipe.get("tags", [])
            tags_text = ", ".join(tags) if tags else ""
            blocks.append(
                "\n".join(
                    [
                        f"Recipe: {recipe.get('name', 'Untitled')}",
                        "Ingredients:",
                        ingredients_text or "- (none)",
                        "Instructions:",
                        steps_text or "(none)",
                        f"Tags: {tags_text}",
                    ]
                )
            )

        prompt = (
            "You are a helpful cooking assistant."
            " Use the user query and the candidate recipes to produce a single"
            " adapted recipe that fits the request. If the query specifies constraints"
            " (dietary, time, equipment), respect them."
            " Provide a recipe title, ingredient list, and step-by-step instructions."
            "\n\n"
            f"User query: {query}\n\n"
            "Candidate recipes:\n\n"
            + "\n\n---\n\n".join(blocks)
            + "\n\nReturn the adapted recipe now."
        )

        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return {"response": content.strip()}

    return generate


def _critic_factory(settings: Settings, critic: ChatOllama):
    def critique(state: GraphState) -> GraphState:
        response = state.get("response", "")
        if not response:
            return {"critique": {"safe": True, "issues": []}}
        prompt = (
            "You are a safety checker for cooking instructions."
            " Identify dangerous or impossible instructions, extreme cook times,"
            " or unsafe temperatures. Return JSON only."
            "\n\n"
            f"Recipe:\n{response}"
        )
        result = critic.invoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"safe": True, "issues": []}
        return {"critique": parsed}

    return critique


def build_graph(top_k: int = DEFAULT_TOP_K):
    settings = load_settings()
    embedder = OllamaEmbeddings(
        model=settings.embed_model,
        base_url=settings.ollama_base_url,
    )
    generator_llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )
    critic_llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        format=CRITIC_JSON_SCHEMA,
    )

    graph = StateGraph(GraphState)
    graph.add_node("retriever", _retrieve_factory(settings, embedder, top_k))
    graph.add_node("generator", _generator_factory(settings, generator_llm))
    graph.add_node("critic", _critic_factory(settings, critic_llm))

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "critic")
    graph.add_edge("critic", END)

    return graph.compile()
