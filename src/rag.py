"""RAG: load policy docs, build FAISS index once at startup. Expose retriever for agents."""
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Smaller/faster embedding model for lower latency. L3 is ~5x faster than L6.
# Override with EMBEDDING_MODEL env if you want higher quality (e.g. all-MiniLM-L6-v2).
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"


def load_policy_docs(policy_docs_dir: Path) -> list[Document]:
    """Load all .md files under policy_docs_dir into LangChain Documents."""
    docs = []
    if not policy_docs_dir.exists():
        return docs
    for path in sorted(policy_docs_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
            docs.append(Document(page_content=text, metadata={"source": path.name}))
        except OSError:
            continue
    return docs


class _EmptyRetriever:
    def invoke(self, query: str):
        return []


def build_retriever(policy_docs_dir: Path, embedding_model: str | None = None, k: int = 3):
    """Build FAISS vector store from policy docs and return a retriever. Call once at startup."""
    if embedding_model is None:
        embedding_model = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    doc_list = load_policy_docs(policy_docs_dir)
    if not doc_list:
        return _EmptyRetriever()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = splitter.split_documents(doc_list)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})
