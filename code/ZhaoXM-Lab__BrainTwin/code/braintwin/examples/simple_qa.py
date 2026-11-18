"""Ask BrainTwin one question.

Place the literature index and embedding model where DATA.md says, and set
BRAINTWIN_LLM_URL, BRAINTWIN_MODEL_ID, and BRAINTWIN_NEO4J_URL. Set
BRAINTWIN_NEO4J_PASSWORD when the question needs the knowledge graph.

Usage:
    python braintwin/examples/simple_qa.py
    python braintwin/examples/simple_qa.py "On which chromosome is SNP rs2569880 located?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multiagent_system.main import create_system


def main():
    question = "What structural MRI finding is reported for the hippocampus in schizophrenia?"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    system = create_system()
    print(system.query(question))


if __name__ == "__main__":
    main()
