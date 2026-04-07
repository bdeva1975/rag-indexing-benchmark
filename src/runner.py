# src/runner.py
# Orchestrates document loading, strategy execution, and result collection.

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from src.strategies import (
    strategy_fixed_chunking,
    strategy_structure_splitting,
    strategy_parent_document,
    strategy_summary_embeddings,
    strategy_hypothetical_questions,
    strategy_chunk_expansion
)
from src.evaluator import evaluate_strategy


def load_documents(config: dict) -> list:
    """Load all supported documents from the data directory."""
    input_dir = config["data"]["input_dir"]
    file_types = config["data"]["file_types"]
    all_docs = []

    files = list(Path(input_dir).iterdir())
    supported = [f for f in files if f.suffix.lower() in file_types]

    if not supported:
        raise ValueError(
            f"No supported documents found in '{input_dir}'.\n"
            f"Supported types: {file_types}\n"
            f"Please add at least one PDF or TXT file to the data/ folder."
        )

    print(f"\n📂 Loading {len(supported)} document(s) from '{input_dir}'...")
    for file_path in supported:
        print(f"   - {file_path.name}")
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"   ⚠️  Could not load {file_path.name}: {e}")

    print(f"   ✅ Loaded {len(all_docs)} page(s) total.\n")
    return all_docs


def run_all_strategies(docs: list, config: dict) -> list:
    """Run all 6 strategies and evaluate each one."""

    strategies = [
        ("1. Fixed Chunking",          strategy_fixed_chunking),
        ("2. Structure Splitting",      strategy_structure_splitting),
        ("3. ParentDocumentRetriever",  strategy_parent_document),
        ("4. Summary Embeddings",       strategy_summary_embeddings),
        ("5. Hypothetical Questions",   strategy_hypothetical_questions),
        ("6. Chunk Expansion",          strategy_chunk_expansion),
    ]

    all_results = []

    for strategy_name, strategy_fn in strategies:
        print(f"\n{'='*55}")
        print(f"🔄 Strategy: {strategy_name}")
        print(f"{'='*55}")
        try:
            retriever = strategy_fn(docs, config)
            results = evaluate_strategy(strategy_name, retriever, config)
            all_results.extend(results)
        except Exception as e:
            print(f"  ❌ Strategy failed: {e}")

    return all_results