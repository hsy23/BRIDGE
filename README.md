# BRIDGE

**B**lock-wise Speculative Decoding for **R**etrieval-Augmented **I**n-situ **D**evice-Cloud **Ge**neration

BRIDGE is a distributed RAG (Retrieval-Augmented Generation) framework designed for device-cloud collaborative inference. It enables the simultaneous integration of **personalized on-device information** and **generic cloud-side knowledge** through a novel block-wise speculative decoding mechanism, significantly reducing generation latency while preserving output quality.

## Overview

In real-world deployment, user-specific data (e.g., chat history, preferences) often resides on-device due to privacy constraints, while rich generic knowledge bases are hosted on the cloud. BRIDGE bridges this gap by:

- Running parallel decoding processes on both device and cloud
- Retrieving personalized context locally and generic knowledge remotely
- Aggregating heterogeneous outputs via a numerically stable logsumexp-based method
- Adopting **block-wise speculative decoding** to amortize communication overhead and improve throughput

<!-- ![Framework Overview](assets/overview.png) -->

## Architecture

```
                  ┌──────────────────────┐
                  │       BRIDGE         │
                  │  (Orchestrator)      │
                  └──────┬───────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    ┌─────▼─────┐  ┌─────▼─────┐  ┌────▼──────┐
    │  Decoder   │  │Aggregator │  │Transceiver│
    │(Decoding)  │  │(Block Agg)│  │ (Comm)    │
    └─────┬──────┘  └───────────┘  └───────────┘
          │
    ┌─────▼─────┐
    │    RAG     │
    │(Retrieval) │
    └──┬─────┬──┘
       │     │
  ┌────▼┐  ┌─▼───────┐
  │Index│  │Generator │
  └─────┘  │  (LLM)  │
           └─────────┘
```

### Core Modules

| Module | Description | Path |
|--------|-------------|------|
| `BRIDGE` | Top-level orchestrator coordinating device-cloud generation | `BRIDGE/bridge.py` |
| `Generator` | Preemptable causal LM wrapper with KV-cache management | `BRIDGE/generator.py` |
| `Retriever` | Pluggable retrieval backends (Contriever, DPR, Remote) | `BRIDGE/retriever.py` |
| `Indexer` | FAISS-based vector indexing with optional PQ compression | `BRIDGE/indexer.py` |
| `Decoder` | Autoregressive decoding thread with scroll-back support | `BRIDGE/decoder.py` |
| `Aggregator` | Output aggregation (synchronized / speculative / block) | `BRIDGE/aggregator.py` |
| `Transceiver` | TCP-based device-cloud communication layer | `BRIDGE/transceiver.py` |
| `DraftQueue` | Manages draft tokens during speculative decoding | `BRIDGE/queues.py` |

## Getting Started

### Prerequisites

- Python >= 3.10
- PyTorch (with CUDA support recommended for cloud-side)
- FAISS

### Installation

```bash
git clone https://github.com/<your-org>/BRIDGE.git
cd BRIDGE
pip install -r requirements.txt
```

### Quick Start

BRIDGE operates with two coordinated processes — one on the **cloud** side and one on the **device** side.

**1. Start the retriever service (cloud)**

```bash
python BRIDGE/toolbox/retriever_as_a_service.py 8765
```

**2. Launch the cloud-side process**

```bash
python experiments/CoGen/eval.py \
  --trans.rank 0 \
  --aggregator.mode BRIDGE \
  --device cuda:0 \
  --generator.model <path-to-model> \
  --generator.use_fp16 \
  --retriever.passages "wikipedia[remote]" \
  --retriever.n_docs 8 \
  --retriever.s_aggregate 1 \
  --retriever.host 127.0.0.1 \
  --retriever.port 8765
```

**3. Launch the device-side process**

```bash
python experiments/CoGen/eval.py \
  --trans.rank 1 \
  --aggregator.mode BRIDGE \
  --device cpu \
  --generator.model <path-to-model> \
  --generator.use_fp16 \
  --retriever.passages "cogen" \
  --retriever.n_docs 0 \
  --retriever.s_aggregate 1
```

> `--aggregator.mode` supports three modes: `synchronized`, `speculative`, and `BRIDGE` (block-wise).

### Configuration

All hyperparameters are managed via `BRIDGEConfig` (`BRIDGE/config.py`). Key options:

| Category | Parameter | Description |
|----------|-----------|-------------|
| Generator | `generator.model` | HuggingFace model name or local path |
| Generator | `generator.use_fp16` | Enable FP16 inference |
| Retriever | `retriever.n_docs` | Number of retrieved passages |
| Retriever | `retriever.s_context` | Max tokens per passage context |
| Aggregator | `aggregator.mode` | `synchronized` / `speculative` / `BRIDGE` |
| Transceiver | `trans.rank` | Node role (`0` = cloud, `1` = device) |
| Sampler | `sampler.do_sample` | Enable stochastic sampling |

## Project Structure

```
BRIDGE/
├── BRIDGE/                     # Core framework
│   ├── bridge.py               # Main orchestrator
│   ├── config.py               # Configuration system
│   ├── rag.py                  # Retrieval-augmented generation
│   ├── generator.py            # LLM wrapper
│   ├── retriever.py            # Retrieval backends
│   ├── indexer.py              # FAISS indexing
│   ├── aggregator.py           # Output aggregation strategies
│   ├── decoder.py              # Decoding with scroll-back
│   ├── transceiver.py          # Network communication
│   ├── queues.py               # Draft token management
│   ├── models/                 # Retriever model definitions
│   ├── baselines/              # Baseline implementations
│   ├── toolbox/                # Utility services
│   └── utils/                  # Logging, caching, text processing
├── experiments/                # Evaluation scripts
│   ├── CoGen/                  # Personalized generation evaluation
│   ├── MovieExp/               # Movie domain evaluation
│   ├── Latency/                # Latency profiling
│   └── LanguageModeling/       # Language modeling evaluation
├── datasets/                   # Evaluation datasets
├── requirements.txt
├── Makefile
└── LICENSE
```

## Citation

If you find BRIDGE useful in your research, please cite our paper:

```bibtex
@inproceedings{bridge2026,
  title     = {BRIDGE: Block-wise Speculative Decoding for Device-Cloud Collaborative RAG},
  author    = {Anonymous},
  booktitle = {KDD},
  year      = {2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
