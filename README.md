# ESG-RAG

**Evidence-aware, Structure-Guided Retrieval-Augmented Generation** for multi-hop question answering.

![ESG-RAG framework](docs/esg-rag-framework.svg)

ESG-RAG keeps only the code needed by the paper's method. It extends GraphRAG 0.4.1 with three stages:

1. **Structure-guided indexing** selects representative and bridge chunks from a semantic k-NN graph while building a lightweight keyword index.
2. **Retrieval-feedback decomposition (RF-DQD)** executes dependency-aware subqueries and accumulates structured and textual evidence separately.
3. **Evidence-aware generation and selection** scores both channel answers, uses cross-channel support, and conditionally fuses complementary evidence.

## Installation

ESG-RAG supports Python 3.10-3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords
```

Poetry users can run `poetry install` instead.

Verify the Python version and installed packages without making any API call:

```bash
python -m esg_rag.preflight
```

## Dataset layout

Create one workspace per dataset:

```text
workspace/my-dataset/
|-- input/                 # source .txt documents
|-- qa-pairs/
|   `-- qa-pairs.json      # [{"id": "...", "question": "...", ...}]
`-- settings.yaml          # generated in the next step
```

Do not commit API keys, datasets, caches, or generated outputs. They are covered by `.gitignore`.

This repository includes the 500-question MuSiQue example used by the original
project under `examples/musique/`. On Windows, the complete example can be run
with a small default limit after installation:

```powershell
$env:OPENAI_API_KEY="your-key"
.\scripts\run_musique_example.ps1 -Limit 5
```

## Reproduce the method

Set the API key first:

```bash
# PowerShell
$env:OPENAI_API_KEY="your-key"
$env:GRAPHRAG_API_KEY=$env:OPENAI_API_KEY
```

Then run the four stages from the repository root:

```bash
# 1. Initialize prompts/configuration, then build the structure-guided index.
python -m graphrag init --root workspace/my-dataset
python -m graphrag index --root workspace/my-dataset

# Optional: verify dataset inputs and all generated index files.
python -m esg_rag.preflight workspace/my-dataset --require-index

# 2. RF-DQD dual-channel retrieval. The optional number limits samples (max 500).
python -m esg_rag.retrieve workspace/my-dataset 500

# 3. Generate one candidate answer per evidence channel.
python -m esg_rag.generate workspace/my-dataset rf-dqd

# 4. Cross-evidence selection and conditional fusion.
python -m esg_rag.select workspace/my-dataset rf-dqd standard
```

The generated `settings.yaml` already uses ESG-RAG's default topology selector (`budget=0.8`, `knn_edges=2`, `split_size=150`). These values can be changed under `chunks` for controlled experiments.

The final predictions are written below the dataset's `output/` directory as `answer-gpt4omini-aligned-dual-channel-cross-evidence-fusion-rf-dqd-raw.json`.

## Configuration

Useful environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | required | Retrieval, answer, and fusion API key |
| `GRAPHRAG_API_KEY` | required for indexing | GraphRAG indexing API key |
| `OPENAI_ANSWER_MODEL` | `gpt-4o-mini` | Candidate answer model |
| `RF_DQD_MODEL` | `gpt-4o-mini` | Query decomposition model |
| `RF_DQD_CONCURRENCY` | `2` | Concurrent retrieval-feedback questions |
| `RF_DQD_STRUCTURAL_TOKENS` | `12000` | Structured-channel context budget |
| `RF_DQD_TEXTUAL_TOKENS` | `12000` | Textual-channel context budget |

Run `python -m esg_rag.select --help` to inspect evidence-scoring weights and thresholds.

## Repository structure

```text
esg_rag/
|-- indexing.py       # structure-guided chunk selection and keyword indexing
|-- topology.py       # representative/bridge selector
|-- decompose.py      # RF-DQD
|-- retrieve.py       # dual-channel retrieval and trace generation
|-- generate.py       # channel-specific answer generation
`-- select.py         # cross-evidence decision and conditional fusion
graphrag/             # GraphRAG 0.4.1 base, adapted for ESG-RAG indexing
tests/                # offline unit and CLI smoke tests
examples/musique/     # bundled 500-question reproducibility example
docs/                 # ESG-RAG framework figure
scripts/              # executable example pipeline
```

## Acknowledgments

The indexing and query foundation is based on [Microsoft GraphRAG 0.4.1](https://github.com/microsoft/graphrag/tree/v0.4.1). See `LICENSE` for licensing information.

