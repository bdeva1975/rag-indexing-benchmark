# benchmark.py
# Main entry point. Run this to start the benchmark.
# Usage: python benchmark.py

import os
import yaml
import pandas as pd
import logging
from tabulate import tabulate
from dotenv import load_dotenv
from src.runner import load_documents, run_all_strategies

# ── Load environment variables ─────────────────────────────────────
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Suppress ChromaDB telemetry warnings
logging.getLogger("chromadb").setLevel(logging.ERROR)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_results(results: list, config: dict):
    """Save detailed results to CSV."""
    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    output_path = os.path.join(
        config["output"]["results_dir"],
        config["output"]["filename"]
    )
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Detailed results saved to: {output_path}")
    return df


def print_leaderboard(df: pd.DataFrame):
    """Print a ranked leaderboard summarizing each strategy."""
    leaderboard = (
        df.groupby("strategy")
        .agg(
            avg_relevance=("relevance_score", "mean"),
            avg_faithfulness=("faithfulness_score", "mean"),
            avg_completeness=("completeness_score", "mean"),
            avg_overall=("overall_score", "mean")
        )
        .round(4)
        .sort_values("avg_overall", ascending=False)
        .reset_index()
    )

    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
    leaderboard.columns = [
        "Rank", "Strategy",
        "Relevance", "Faithfulness", "Completeness", "Overall"
    ]

    print("\n")
    print("=" * 65)
    print("🏆  RAG INDEXING BENCHMARK — LEADERBOARD")
    print("=" * 65)
    print(tabulate(
        leaderboard,
        headers="keys",
        tablefmt="rounded_outline",
        showindex=False
    ))
    print("=" * 65)
    print(f"\n🥇 Best strategy: {leaderboard.iloc[0]['Strategy']}")
    print(f"   Overall score: {leaderboard.iloc[0]['Overall']}")
    print("\n")


def main():
    print("\n🚀 RAG Indexing Benchmark Starting...")
    print("─" * 55)

    config = load_config()
    docs = load_documents(config)
    results = run_all_strategies(docs, config)
    df = save_results(results, config)
    print_leaderboard(df)

    print("✅ Benchmark complete.\n")


if __name__ == "__main__":
    main()