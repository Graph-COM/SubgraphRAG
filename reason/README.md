# Stage 2: Reasoning

## Installation

```bash
conda create -n reasoner python=3.10.14 -y
conda activate reasoner
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.5.5
pip install openai==1.50.2
```

## Our results

We provide our retriever's results in `./scored_triples` and our reasoning results in `./results`. Please run the following command to download all our results.

```
huggingface-cli download siqim311/SubgraphRAG --revision main --local-dir ./
```

## Inference with LLMs

First put the retriver's results in `./scored_triples`. Then, one can run `main.py` with proper paramerters. For example,

```
python main.py -d webqsp --prompt_mode scored_100
```

Our used config for each dataset can be found in `./config`.

