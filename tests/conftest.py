from pathlib import Path

import pytest

from small.corpus import load_jsonl

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CORPORA_DIR = FIXTURES_DIR / "corpora"
BENCHMARKS_DIR = FIXTURES_DIR / "benchmarks"


@pytest.fixture
def python_small_corpus():
    return load_jsonl(CORPORA_DIR / "code" / "python_small.jsonl")


@pytest.fixture
def security_policies_corpus():
    return load_jsonl(CORPORA_DIR / "policies" / "security_policies.jsonl")


@pytest.fixture
def edge_case_minimal():
    return load_jsonl(CORPORA_DIR / "edge_cases" / "minimal_repetition.jsonl")


@pytest.fixture
def edge_case_maximal():
    return load_jsonl(CORPORA_DIR / "edge_cases" / "maximum_repetition.jsonl")


@pytest.fixture
def security_qa_benchmark():
    return load_jsonl(BENCHMARKS_DIR / "security_qa.jsonl")


@pytest.fixture(params=["python_small", "typescript_small", "security_policies"])
def sample_corpus(request):
    corpus_paths = {
        "python_small": CORPORA_DIR / "code" / "python_small.jsonl",
        "typescript_small": CORPORA_DIR / "code" / "typescript_small.jsonl",
        "security_policies": CORPORA_DIR / "policies" / "security_policies.jsonl",
    }
    return load_jsonl(corpus_paths[request.param])
