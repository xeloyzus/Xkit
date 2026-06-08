"""Tests for the retrieval module."""

from xkit.retrieval import LocalTfidfRetriever, tokenize, create_retriever


def test_tokenize():
    """Tokenization should extract words and lowercase them."""
    assert tokenize("Hello World") == ["hello", "world"]
    assert tokenize("foo_bar baz") == ["foo_bar", "baz"]
    assert tokenize("") == []
    assert tokenize("123") == []


def test_tfidf_search_basic():
    """TF-IDF should find the most relevant chunk."""
    chunks = [
        {"text": "def login(): pass", "file": "auth.py", "symbol": "login"},
        {"text": "def logout(): pass", "file": "auth.py", "symbol": "logout"},
        {"text": "def format_date(): pass", "file": "utils.py", "symbol": "format_date"},
    ]
    retriever = LocalTfidfRetriever(chunks)
    results = retriever.search("login authentication", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk["symbol"] == "login"


def test_tfidf_search_top_k():
    """top_k should limit results."""
    chunks = [
        {"text": f"function_{i}(): pass", "file": f"file_{i}.py", "symbol": f"func_{i}"}
        for i in range(20)
    ]
    retriever = LocalTfidfRetriever(chunks)
    results = retriever.search("function", top_k=5)
    assert len(results) <= 5


def test_tfidf_empty_query():
    """An empty query should return no results."""
    chunks = [
        {"text": "def foo(): pass", "file": "test.py", "symbol": "foo"},
    ]
    retriever = LocalTfidfRetriever(chunks)
    results = retriever.search("", top_k=5)
    assert len(results) == 0


def test_tfidf_no_match():
    """A query with no matching terms should return no results."""
    chunks = [
        {"text": "def foo(): pass", "file": "test.py", "symbol": "foo"},
    ]
    retriever = LocalTfidfRetriever(chunks)
    results = retriever.search("xyznonexistent123", top_k=5)
    assert len(results) == 0


def test_create_retriever_default():
    """create_retriever with default method should return TF-IDF."""
    chunks = [{"text": "test", "file": "test.py"}]
    retriever = create_retriever(chunks)
    from xkit.retrieval import LocalTfidfRetriever
    assert isinstance(retriever, LocalTfidfRetriever)


def test_create_retriever_embeddings_fallback():
    """create_retriever with 'embeddings' should fall back to TF-IDF if deps missing."""
    chunks = [{"text": "test", "file": "test.py"}]
    retriever = create_retriever(chunks, method="embeddings")
    from xkit.retrieval import LocalTfidfRetriever
    assert isinstance(retriever, LocalTfidfRetriever)
