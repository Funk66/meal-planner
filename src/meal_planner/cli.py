from __future__ import annotations

import typer

from .agent.graph import build_graph

app = typer.Typer(help="Recipe RAG CLI", invoke_without_command=True)


def _run_query(query: str, top_k: int) -> None:
    graph = build_graph(top_k=top_k)

    state: dict = {}
    for update in graph.stream({"query": query}, stream_mode="updates"):
        for node, value in update.items():
            if node == "retriever":
                print("Thinking... retrieving recipes")
            elif node == "generator":
                print("Thinking... generating recipe")
            elif node == "critic":
                print("Thinking... checking for safety")
            if isinstance(value, dict):
                state.update(value)

    response = state.get("response", "")
    critique = state.get("critique", {})

    if response:
        print("\n" + response)
    else:
        print("\nNo response generated.")

    if isinstance(critique, dict) and not critique.get("safe", True):
        issues = critique.get("issues", [])
        if issues:
            print("\nSafety check flagged issues:")
            for issue in issues:
                print(f"- {issue}")


@app.callback()
def main(
    ctx: typer.Context,
    query: str = typer.Argument(None),
    top_k: int = typer.Option(5, help="Number of recipes to retrieve"),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if not query:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)
    _run_query(query, top_k)


@app.command()
def ask(
    query: str,
    top_k: int = typer.Option(5, help="Number of recipes to retrieve"),
) -> None:
    """Ask for recipe ideas using the RAG pipeline."""
    _run_query(query, top_k)


if __name__ == "__main__":
    app()
