"""Optional helper: download a small, text-extractable PDF corpus into data/raw/.

Replace the URLs with documents from YOUR chosen domain. Every PDF listed here is
openly licensed / public. Usage:  python scripts/fetch_corpus.py
"""

from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# domain: machine-learning / NLP literature (arXiv, open access)
SOURCES = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762",
    "bert.pdf": "https://arxiv.org/pdf/1810.04805",
    "rag_knowledge_intensive_nlp.pdf": "https://arxiv.org/pdf/2005.11401",
    "dense_passage_retrieval.pdf": "https://arxiv.org/pdf/2004.04906",
    "sentence_bert.pdf": "https://arxiv.org/pdf/1908.10084",
    "lost_in_the_middle.pdf": "https://arxiv.org/pdf/2307.03172",
    "self_rag.pdf": "https://arxiv.org/pdf/2310.11511",
    "ragas_evaluation.pdf": "https://arxiv.org/pdf/2309.15217",
}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        target = RAW_DIR / name
        if target.exists():
            print(f"skip   {name} (already downloaded)")
            continue
        print(f"fetch  {name} <- {url}")
        response = requests.get(url, timeout=60, headers={"User-Agent": "rag-assistant-project/1.0"})
        response.raise_for_status()
        target.write_bytes(response.content)
    print(f"\nDone. Corpus in {RAW_DIR}")


if __name__ == "__main__":
    main()
