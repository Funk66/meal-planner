from __future__ import annotations

import ast
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pgvector.psycopg import register_vector
from psycopg import connect
from psycopg.types.json import Json
from pydantic import BaseModel, Field
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "qwen2.5:7b"

INGREDIENT_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "search_term": {"type": "string"},
            "quantity": {"type": "number"},
            "unit": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["search_term", "quantity", "unit", "notes"],
        "additionalProperties": False,
    },
}

FEW_SHOT_PROMPT = """\
You are cleaning recipe ingredients into structured data.
Return ONLY a JSON array that matches the provided schema.

Rules:
- search_term: ingredient name only (no quantities or units).
- quantity: decimal number (use 0.5 for 1/2).
- unit: unit name, or empty string if missing.
- notes: preparation notes or descriptors not part of the search_term (e.g., "melted").

Example 1
Input:
["1/2 cup unsalted butter, melted", "2 cups mashed canned Louisiana yams"]
Output:
[
  {"search_term": "unsalted butter", "quantity": 0.5, "unit": "cup", "notes": "melted"},
  {"search_term": "louisiana yams", "quantity": 2.0, "unit": "cups", "notes": "mashed, canned"}
]

Example 2
Input:
["3 cloves garlic, minced", "1 tbsp olive oil"]
Output:
[
  {"search_term": "garlic", "quantity": 3.0, "unit": "cloves", "notes": "minced"},
  {"search_term": "olive oil", "quantity": 1.0, "unit": "tbsp", "notes": ""}
]
"""

MODEL_PULL_STATE: dict[str, bool] = {}


class Ingredient(BaseModel):
    search_term: str
    quantity: float
    unit: str
    notes: str


class Recipe(BaseModel):
    recipe_id: int
    name: str
    ingredients: list[Ingredient] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)


@dataclass
class Settings:
    postgres_connection_string: str | None
    ollama_base_url: str
    embed_model: str
    llm_model: str


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        postgres_connection_string=os.getenv("POSTGRES_CONNECTION_STRING"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        embed_model=os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        llm_model=os.getenv("OLLAMA_LLM_MODEL", DEFAULT_LLM_MODEL),
    )


def _find_local_csv() -> Path | None:
    if not DATA_DIR.exists():
        return None
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    return csv_files[0] if csv_files else None


def _parse_list_field(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except ValueError, SyntaxError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        delimiter = "\n" if "\n" in text else ","
        return [part.strip() for part in text.split(delimiter) if part.strip()]
    return [str(value).strip()]


def _extract_json_array(text: str) -> list[dict] | None:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return parsed or None


def _is_model_not_found_error(error: Exception) -> bool:
    message = str(error).lower()
    if "not found" in message and "model" in message:
        return True
    if "pull" in message and "model" in message:
        return True
    return False


def _prompt_pull_model(model_name: str) -> bool:
    response = input(
        f"Ollama model '{model_name}' is not available. Pull it now? [y/N]: "
    )
    return response.strip().lower() in {"y", "yes"}


def _ollama_request(base_url: str, path: str, payload: dict) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        body = response.read().decode("utf-8")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unexpected Ollama response: {body}") from exc


def _pull_ollama_model(base_url: str, model_name: str) -> None:
    try:
        response = _ollama_request(
            base_url,
            "/api/pull",
            {"name": model_name, "stream": False},
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Failed to pull model '{model_name}'. Ollama responded with: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to reach Ollama at {base_url}. Is it running?"
        ) from exc

    if response.get("error"):
        raise RuntimeError(
            f"Failed to pull model '{model_name}': {response.get('error')}"
        )


def _call_with_model_retry(
    action_name: str,
    model_name: str,
    base_url: str,
    action,
):
    try:
        return action()
    except Exception as exc:  # noqa: BLE001
        if not _is_model_not_found_error(exc):
            raise
        if model_name in MODEL_PULL_STATE and not MODEL_PULL_STATE[model_name]:
            raise RuntimeError(
                f"Model '{model_name}' is required for {action_name}, but was not pulled."
            ) from exc
        if not _prompt_pull_model(model_name):
            MODEL_PULL_STATE[model_name] = False
            raise RuntimeError(
                f"Model '{model_name}' is required for {action_name}, but was not pulled."
            ) from exc
        _pull_ollama_model(base_url, model_name)
        MODEL_PULL_STATE[model_name] = True
        return action()


def _coerce_ingredient(item: dict) -> Ingredient | None:
    search_term = str(item.get("search_term", "")).strip()
    if not search_term:
        return None
    quantity_raw = item.get("quantity") or 1
    try:
        quantity = float(quantity_raw)
    except TypeError, ValueError:
        return None
    unit = str(item.get("unit", "")).strip()
    notes = str(item.get("notes", "")).strip()
    return Ingredient(
        search_term=search_term, quantity=quantity, unit=unit, notes=notes
    )


def _clean_ingredients(
    llm: ChatOllama,
    raw_items: list[str],
    base_url: str,
    model_name: str,
) -> list[Ingredient]:
    if not raw_items:
        return []
    prompt = f"{FEW_SHOT_PROMPT}\nInput:\n{json.dumps(raw_items, ensure_ascii=False)}"
    response = _call_with_model_retry(
        action_name="ingredient parsing",
        model_name=model_name,
        base_url=base_url,
        action=lambda: llm.invoke(prompt),
    )
    content = str(response.content) if hasattr(response, "content") else str(response)
    parsed = _extract_json_array(content)
    if not parsed:
        return []
    cleaned: list[Ingredient] = []
    for item in parsed:
        ingredient = _coerce_ingredient(item)
        if ingredient is None:
            return []
        cleaned.append(ingredient)
    return cleaned


def _prepare_recipe(
    row: pd.Series,
    llm: ChatOllama,
    embedder: OllamaEmbeddings,
    base_url: str,
    llm_model: str,
    embed_model: str,
) -> Recipe | None:
    recipe_id = int(row.get("id") or 0)
    name = str(row.get("name") or "").strip()
    raw_source = row.get("ingredients_raw")
    raw_items = _parse_list_field(raw_source)
    if not raw_items:
        raw_items = _parse_list_field(row.get("ingredients"))
    cleaned_ingredients = _clean_ingredients(llm, raw_items, base_url, llm_model)
    if not cleaned_ingredients:
        return None

    instructions = _parse_list_field(row.get("steps"))
    tags = _parse_list_field(row.get("tags"))

    embed_terms = [ingredient.search_term for ingredient in cleaned_ingredients]
    embed_text = "\n".join(term for term in embed_terms if term)
    if embed_text:
        embedding = _call_with_model_retry(
            action_name="embedding generation",
            model_name=embed_model,
            base_url=base_url,
            action=lambda: embedder.embed_query(embed_text),
        )
    else:
        embedding = []

    return Recipe(
        recipe_id=recipe_id,
        name=name or "(untitled)",
        ingredients=cleaned_ingredients,
        instructions=instructions,
        tags=tags,
        embedding=embedding,
    )


def _ensure_schema(conn) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recipes (
            id BIGINT PRIMARY KEY,
            name TEXT NOT NULL,
            ingredients JSONB NOT NULL,
            instructions JSONB NOT NULL,
            tags JSONB NOT NULL,
            embedding VECTOR(768) NOT NULL
        )
        """
    )


def _fetch_existing_ids(conn) -> set[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM recipes")
        return {row[0] for row in cur.fetchall()}


def _insert_recipe(conn, recipe: Recipe) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO recipes (id, name, ingredients, instructions, tags, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                recipe.recipe_id,
                recipe.name,
                Json([ingredient.model_dump() for ingredient in recipe.ingredients]),
                Json(recipe.instructions),
                Json(recipe.tags),
                recipe.embedding,
            ),
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Process recipes and store in Postgres"
    )
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = load_settings()

    csv_path = args.csv_path or _find_local_csv()
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(
            "No CSV found. Place a dataset CSV in data/ or pass --csv-path."
        )

    limit = 5 if args.dry_run else args.limit
    df = pd.read_csv(csv_path)
    if limit is not None and limit > 0:
        df = df.head(limit)

    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        format=INGREDIENT_JSON_SCHEMA,
    )
    embedder = OllamaEmbeddings(
        model=settings.embed_model, base_url=settings.ollama_base_url
    )

    if args.dry_run:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing recipes"):
            recipe = _prepare_recipe(
                row,
                llm,
                embedder,
                settings.ollama_base_url,
                settings.llm_model,
                settings.embed_model,
            )
            if recipe is not None:
                print(recipe.model_dump())
        return

    if not settings.postgres_connection_string:
        raise RuntimeError("POSTGRES_CONNECTION_STRING is not set in the environment")

    with connect(settings.postgres_connection_string) as conn:
        _ensure_schema(conn)
        conn.commit()
        register_vector(conn)
        existing_ids = _fetch_existing_ids(conn)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing recipes"):
            recipe_id = int(row.get("id") or 0)
            if recipe_id in existing_ids:
                continue
            recipe = _prepare_recipe(
                row,
                llm,
                embedder,
                settings.ollama_base_url,
                settings.llm_model,
                settings.embed_model,
            )
            if recipe is not None:
                _insert_recipe(conn, recipe)
                existing_ids.add(recipe.recipe_id)


if __name__ == "__main__":
    main()
