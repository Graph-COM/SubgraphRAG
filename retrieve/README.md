# Stage 1: Retrieval

## Table of Contents

- [1-1 Embedding Pre-Computation](#1-1-entity-and-relation-embedding-pre-computation)
    * [Installation](#installation)

## 1-1: Entity and Relation Embedding Pre-Computation

We first pre-compute and cache entity and relation embeddings for all samples to save time for later training and inference of retrievers.

### Installation

```bash
conda create -n gte_large_en_v1-5 python=3.10 -y
conda activate gte_large_en_v1-5
pip install -r gte_large_en_v1-5_requirements.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```
