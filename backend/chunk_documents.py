"""Split LangChain resume documents using the embedding model's token counts."""

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_MAX_TOKENS = 256
CHUNK_SIZE = 220
CHUNK_OVERLAP = 30


def split_resume_documents(
    documents: list[Document],
    *,
    tokenizer: Any = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Return token-bounded chunks with stable IDs and source metadata.

    Prefer paragraph, line, then word boundaries. Short courses and skills
    remain single chunks. Text is not reduced to keywords or normalized.
    Overlap is a target: natural boundaries can produce less overlap.
    The optional tokenizer lets callers reuse a tokenizer or run offline tests.
    """
    if not documents:
        return []
    if any(not document.id for document in documents):
        raise ValueError("Every source document must have an ID before splitting.")
    if len({document.id for document in documents}) != len(documents):
        raise ValueError("Source document IDs must be unique.")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    if not 0 < chunk_size <= MODEL_MAX_TOKENS - special_tokens:
        raise ValueError("chunk_size must leave room for the model's special tokens.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be nonnegative and smaller than chunk_size.")

    def token_count(text: str) -> int:
        return len(tokenizer.encode(
            text, add_special_tokens=False, truncation=False, verbose=False
        ))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_count,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True,
    )
    chunks = []
    for document in documents:
        parts = splitter.split_documents([document])
        for index, part in enumerate(parts):
            # Measure the assembled chunk too: token counts are not always
            # additive across boundaries. Never silently truncate an input.
            actual_tokens = len(tokenizer.encode(
                part.page_content, add_special_tokens=True,
                truncation=False, verbose=False,
            ))
            if actual_tokens > MODEL_MAX_TOKENS:
                raise ValueError(
                    f"Chunk {index} of {document.id} exceeds the model token limit; "
                    "reduce chunk_size."
                )
            part.id = f"{document.id}:chunk:{index}"
            part.metadata.update(
                parent_id=str(document.id),
                chunk_index=index,
                chunk_count=len(parts),
                token_count=actual_tokens,
            )
            chunks.append(part)
    return chunks
