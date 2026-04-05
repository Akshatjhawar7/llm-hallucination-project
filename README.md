# LLM Hallucination Reduction Project

A comparative study of techniques to reduce hallucinations in large language models, evaluated on the [TruthfulQA](https://github.com/sylinrl/TruthfulQA) benchmark.

**Course:** COMP 4900 — Carleton University

## Overview

LLMs often generate confident but factually incorrect answers (hallucinations). This project implements and compares five hallucination mitigation techniques, measuring their effectiveness at producing truthful answers or appropriately abstaining with "I don't know."

## Techniques

| # | Method | Description | LLM Calls/Question |
|---|--------|-------------|---------------------|
| 1 | **Baseline** | Direct prompting with no constraints | 1 |
| 2 | **Prompt Constraints** | Structured rules enforcing short, factual answers with IDK fallback | 1 |
| 3 | **Self-Correction** | Generate → critique for factual risk → revise based on critique | 3 |
| 4 | **Self-Consistency** | Sample 5 diverse responses (temp=0.8), return consensus via pairwise token F1 agreement, or IDK if no consensus (threshold=0.4) | 5 |
| 5 | **RAG** | Retrieve Wikipedia context, ground the answer in retrieved evidence only | 1 + retrieval |

## Results (Ollama — llama3.1:8b, 817 questions)

| Metric | Baseline | Prompt Constraints | Self-Correction | Self-Consistency | RAG |
|--------|----------|-------------------|-----------------|-----------------|-----|
| Hallucination Rate | 86.5% | 21.8% | 21.4% | **6.9%** | 25.0% |
| IDK Rate | 0.2% | 44.2% | 40.5% | 72.6% | 50.1% |
| Acceptable Rate | 2.2% | 7.5% | **8.9%** | 8.8% | 2.3% |
| Partial Rate | 11.0% | 26.6% | **29.1%** | 11.8% | 22.6% |
| Avg F1 | 0.121 | 0.182 | **0.186** | 0.129 | 0.186 |
| Avg Latency (s) | 20.1 | **3.1** | 10.8 | 13.9 | 3.7 |

**Key findings:**
- All techniques significantly reduce hallucination compared to baseline (86.5%)
- Self-consistency achieves the lowest hallucination rate (6.9%) but at the cost of high IDK rate (72.6%)
- Self-correction offers the best balance between hallucination reduction and useful answers
- A clear tradeoff exists: lower hallucination rate correlates with higher IDK rate

## Project Structure

```
llm-hallucination-project/
├── main.py                      # Interactive menu to run individual methods
├── run_all.py                   # Run all 5 methods and print comparison table
├── methods/
│   ├── baseline.py              # Baseline direct prompting
│   ├── prompt_constraints.py    # Prompt engineering with strict rules
│   ├── self_correction.py       # Generate → critique → revise pipeline
│   ├── self_consistency.py      # Multi-sample consensus via token F1
│   └── rag.py                   # Wikipedia-grounded retrieval augmented generation
├── evaluation/
│   ├── runner.py                # Experiment loop over TruthfulQA
│   ├── metrics.py               # Exact match, token F1, IDK detection, classification
│   └── logger.py                # Save results to JSON/CSV
├── data/
│   └── loader.py                # TruthfulQA dataset loader from HuggingFace
├── utils/
│   ├── llm.py                   # Unified LLM wrapper (Ollama + OpenAI)
│   └── config.py                # Environment-based configuration
├── results/                     # Generated experiment results (JSON + CSV)
├── requirements.txt
└── .env                         # LLM provider config (not tracked)
```

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd llm-hallucination-project
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure `.env`:
   ```env
   # For Ollama (local)
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama3.1:8b
   OLLAMA_URL=http://localhost:11434/api/generate

   # For OpenAI
   # LLM_PROVIDER=openai
   # OPENAI_API_KEY=sk-...
   # OPENAI_MODEL=gpt-4o-mini
   ```

5. If using Ollama, install and start it:
   ```bash
   ollama serve
   ollama pull llama3.1:8b
   ```

## Usage

**Run a single method interactively:**
```bash
python main.py
```

**Run all methods and get a comparison table:**
```bash
python run_all.py
```

Results are saved to the `results/` directory as JSON and CSV files.

## Evaluation Metrics

- **Hallucination Rate** — fraction of responses classified as likely hallucinations (token F1 < 0.2 vs reference)
- **IDK Rate** — fraction of responses where the model abstains ("I don't know")
- **Acceptable Rate** — exact match or token F1 >= 0.5
- **Partial Rate** — token F1 between 0.2 and 0.5
- **Avg F1** — average token-level F1 score against reference answers
- **Avg Latency** — average wall-clock time per question (includes all LLM calls)
