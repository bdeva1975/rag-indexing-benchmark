# src/strategies.py
# Each function ingests documents and returns a retriever.

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore, InMemoryByteStore
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List
import uuid


def get_embeddings(config: dict) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config["embeddings"]["model"]
    )


def get_chroma_collection(name: str, config: dict) -> Chroma:
    return Chroma(
        collection_name=name,
        embedding_function=get_embeddings(config),
        persist_directory=config["chroma"]["persist_dir"]
    )


# ── Strategy 1: Fixed-size chunking (baseline) ─────────────────────
def strategy_fixed_chunking(docs: list, config: dict):
    print("  → Running Strategy 1: Fixed-size chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["fixed"]["chunk_size"],
        chunk_overlap=config["chunking"]["fixed"]["chunk_overlap"]
    )
    chunks = splitter.split_documents(docs)
    collection = get_chroma_collection("strategy_fixed", config)
    collection.reset_collection()
    collection.add_documents(chunks)
    return collection.as_retriever(
        search_kwargs={"k": config["benchmark"]["top_k"]}
    )


# ── Strategy 2: Structure-based splitting (by title/section) ───────
def strategy_structure_splitting(docs: list, config: dict):
    print("  → Running Strategy 2: Structure-based splitting...")
    # Uses smaller chunks to respect natural document boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    collection = get_chroma_collection("strategy_structure", config)
    collection.reset_collection()
    collection.add_documents(chunks)
    return collection.as_retriever(
        search_kwargs={"k": config["benchmark"]["top_k"]}
    )


# ── Strategy 3: ParentDocumentRetriever ────────────────────────────
def strategy_parent_document(docs: list, config: dict):
    print("  → Running Strategy 3: ParentDocumentRetriever...")
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["parent"]["parent_chunk_size"]
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["parent"]["child_chunk_size"]
    )
    child_collection = get_chroma_collection(
        "strategy_parent_child", config
    )
    child_collection.reset_collection()
    doc_store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=child_collection,
        docstore=doc_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    retriever.add_documents(docs, ids=None)
    return retriever


# ── Strategy 4: MultiVectorRetriever with summary embeddings ───────
def strategy_summary_embeddings(docs: list, config: dict):
    print("  → Running Strategy 4: Summary embeddings...")
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["parent"]["parent_chunk_size"]
    )
    coarse_chunks = parent_splitter.split_documents(docs)
    coarse_chunk_ids = [str(uuid.uuid4()) for _ in coarse_chunks]

    llm = ChatOpenAI(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"]
    )
    summarization_chain = (
        {"document": lambda x: x.page_content}
        | ChatPromptTemplate.from_template(
            "Summarize the following document in 3-5 sentences:\n\n{document}"
        )
        | llm
        | StrOutputParser()
    )

    summaries_collection = get_chroma_collection(
        "strategy_summaries", config
    )
    summaries_collection.reset_collection()
    doc_byte_store = InMemoryByteStore()
    doc_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=summaries_collection,
        byte_store=doc_byte_store,
        id_key=doc_key
    )

    all_summaries = []
    for i, chunk in enumerate(coarse_chunks):
        summary_text = summarization_chain.invoke(chunk)
        summary_doc = Document(
            page_content=summary_text,
            metadata={doc_key: coarse_chunk_ids[i]}
        )
        all_summaries.append(summary_doc)

    retriever.vectorstore.add_documents(all_summaries)
    retriever.docstore.mset(
        list(zip(coarse_chunk_ids, coarse_chunks))
    )
    return retriever


# ── Strategy 5: MultiVectorRetriever with hypothetical questions ───
class HypotheticalQuestions(BaseModel):
    questions: List[str] = Field(
        ..., description="List of hypothetical questions"
    )


def strategy_hypothetical_questions(docs: list, config: dict):
    print("  → Running Strategy 5: Hypothetical questions...")
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["parent"]["parent_chunk_size"]
    )
    coarse_chunks = parent_splitter.split_documents(docs)
    coarse_chunk_ids = [str(uuid.uuid4()) for _ in coarse_chunks]

    llm = ChatOpenAI(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"]
    ).with_structured_output(HypotheticalQuestions)

    questions_chain = (
        {"document_text": lambda x: x.page_content}
        | ChatPromptTemplate.from_template(
            "Generate exactly 3 hypothetical questions that "
            "the following text could be used to answer:\n\n{document_text}"
        )
        | llm
        | (lambda x: x.questions)
    )

    questions_collection = get_chroma_collection(
        "strategy_hypothetical", config
    )
    questions_collection.reset_collection()
    doc_byte_store = InMemoryByteStore()
    doc_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=questions_collection,
        byte_store=doc_byte_store,
        id_key=doc_key
    )

    all_question_docs = []
    for i, chunk in enumerate(coarse_chunks):
        questions = questions_chain.invoke(chunk)
        for q in questions:
            all_question_docs.append(
                Document(
                    page_content=q,
                    metadata={doc_key: coarse_chunk_ids[i]}
                )
            )

    retriever.vectorstore.add_documents(all_question_docs)
    retriever.docstore.mset(
        list(zip(coarse_chunk_ids, coarse_chunks))
    )
    return retriever


# ── Strategy 6: Granular chunk expansion ───────────────────────────
def strategy_chunk_expansion(docs: list, config: dict):
    print("  → Running Strategy 6: Chunk expansion...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["expansion"]["chunk_size"]
    )
    granular_chunks = splitter.split_documents(docs)

    expanded_store_items = []
    for i, chunk in enumerate(granular_chunks):
        prev = granular_chunks[i-1].page_content if i > 0 else ""
        curr = chunk.page_content
        nxt = granular_chunks[i+1].page_content \
            if i < len(granular_chunks) - 1 else ""
        expanded_text = "\n".join(
            filter(None, [prev, curr, nxt])
        )
        expanded_id = str(uuid.uuid4())
        chunk.metadata["doc_id"] = expanded_id
        expanded_store_items.append(
            (expanded_id, Document(page_content=expanded_text))
        )

    granular_collection = get_chroma_collection(
        "strategy_expansion", config
    )
    granular_collection.reset_collection()
    doc_byte_store = InMemoryByteStore()

    retriever = MultiVectorRetriever(
        vectorstore=granular_collection,
        byte_store=doc_byte_store,
        id_key="doc_id"
    )
    retriever.vectorstore.add_documents(granular_chunks)
    retriever.docstore.mset(expanded_store_items)
    return retriever