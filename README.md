```markdown
# 🏆 RAG Indexing Benchmark

> **Compare 6 RAG indexing strategies on your own documents — with a single command.**

Most RAG tutorials show you *how* to implement indexing strategies. This repo answers the question nobody else does: **which strategy actually performs best on your data?**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3.25-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.0.9-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What It Does

Drop your documents into the `data/` folder, run one command, and get a ranked leaderboard showing which RAG indexing strategy retrieves the most relevant, faithful, and complete answers for your specific content.

```bash
python benchmark.py
```

```
🏆  RAG INDEXING BENCHMARK — LEADERBOARD
=================================================================
│ Rank │ Strategy                   │ Relevance │ Faithfulness │ Completeness │ Overall │
│    1 │ 2. Structure Splitting     │    0.3651 │       1.0000 │       1.0000 │  0.7884 │
│    2 │ 4. Summary Embeddings      │    0.3556 │       1.0000 │       1.0000 │  0.7852 │
│    3 │ 1. Fixed Chunking          │    0.3394 │       1.0000 │       1.0000 │  0.7798 │
│    4 │ 6. Chunk Expansion         │    0.3220 │       1.0000 │       1.0000 │  0.7740 │
│    5 │ 3. ParentDocumentRetriever │    0.3301 │       0.9333 │       1.0000 │  0.7545 │
│    6 │ 5. Hypothetical Questions  │    0.2964 │       0.8667 │       1.0000 │  0.7210 │
=================================================================
🥇 Best strategy: 2. Structure Splitting
```

---

## 🔬 The 6 Strategies

| # | Strategy | Best For |
|---|----------|----------|
| 1 | **Fixed-size Chunking** | Baseline — unstructured prose |
| 2 | **Structure-based Splitting** | HTML, Markdown, section-heavy docs |
| 3 | **ParentDocumentRetriever** | General purpose — balances precision and context |
| 4 | **Summary Embeddings** | Dense factual content — research papers, reports |
| 5 | **Hypothetical Questions** | Complex queries — query-answer vocabulary mismatch |
| 6 | **Chunk Expansion** | Narrative docs — sequential context matters |

---

## 📊 Scoring Dimensions

Each strategy is scored across 3 dimensions per question:

- **Relevance** — Cosine similarity between the question embedding and retrieved context embedding
- **Faithfulness** — LLM-judged score: is the answer grounded in the retrieved context?
- **Completeness** — LLM-judged score: does the answer fully address the question?
- **Overall** — Average of the three scores above

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/bdeva1975/rag-indexing-benchmark.git
cd rag-indexing-benchmark
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key
Create a `.env` file in the root folder:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Add your documents
Drop any `.pdf` or `.txt` files into the `data/` folder.

### 6. Configure your benchmark questions
Edit `config.yaml` and add questions relevant to your documents:
```yaml
benchmark:
  questions:
    - "What are the main topics covered in the document?"
    - "Summarize the key findings."
    - "What recommendations are provided?"
```

### 7. Run the benchmark
```bash
python benchmark.py
```

Results are saved to `results/benchmark_results.csv`.

---

## ⚙️ Configuration

All settings are in `config.yaml`:

```yaml
llm:
  model: "gpt-4o-mini"        # change to gpt-4o for higher quality
  temperature: 0
  max_tokens: 500

embeddings:
  model: "text-embedding-3-small"

chunking:
  fixed:
    chunk_size: 1000
    chunk_overlap: 100
  parent:
    parent_chunk_size: 3000
    child_chunk_size: 500
  expansion:
    chunk_size: 500

benchmark:
  top_k: 3                    # number of chunks retrieved per query
```

---

## 📁 Project Structure

```
rag-indexing-benchmark/
│
├── src/
│   ├── strategies.py     # All 6 indexing strategy implementations
│   ├── evaluator.py      # Scoring: relevance, faithfulness, completeness
│   └── runner.py         # Document loader and strategy orchestrator
│
├── data/                 # ← Put your documents here
├── results/              # ← Benchmark CSV output saved here
│
├── benchmark.py          # Main entry point
├── config.yaml           # All settings
└── requirements.txt      # Dependencies
```

---

## 💡 When to Use Each Strategy

**Your document has clear headings/sections?**
→ Start with Strategy 2 (Structure Splitting)

**Your document is dense with facts and numbers?**
→ Try Strategy 4 (Summary Embeddings)

**Your queries are complex or abstract?**
→ Try Strategy 5 (Hypothetical Questions)

**You need a safe, general-purpose baseline?**
→ Strategy 3 (ParentDocumentRetriever)

**You care about narrative continuity?**
→ Strategy 6 (Chunk Expansion)

---

## 🔗 Related Projects

- [HallucinationBench](https://github.com/bdeva1975/HallucinationBench) — RAG hallucination detection library

---

## 📖 Based On

Concepts and techniques from:
> *AI Agents and Applications with LangChain, LangGraph and MCP* — Roberto Infante (Manning, 2026)
> Chapters 8 & 9: Advanced Indexing and Question Transformations

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*If this repo helped you, please consider giving it a ⭐ — it helps others find it.*
```