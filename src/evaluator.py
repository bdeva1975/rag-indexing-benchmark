# src/evaluator.py
# Evaluates each strategy by asking benchmark questions
# and scoring the retrieved context + generated answer.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
import numpy as np


def get_answer(retriever, question: str, config: dict) -> tuple:
    """Retrieve context and generate an answer for a question."""
    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Generate answer using LLM
    llm = ChatOpenAI(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"]
    )
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based only on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return answer, context, retrieved_docs


def score_relevance(question: str, context: str, config: dict) -> float:
    """
    Score how relevant the retrieved context is to the question
    using cosine similarity between their embeddings.
    """
    embeddings = OpenAIEmbeddings(model=config["embeddings"]["model"])
    q_vec = embeddings.embed_query(question)
    c_vec = embeddings.embed_query(context[:2000])  # trim for cost

    q = np.array(q_vec)
    c = np.array(c_vec)
    cosine_sim = np.dot(q, c) / (np.linalg.norm(q) * np.linalg.norm(c))
    return round(float(cosine_sim), 4)


def score_faithfulness(answer: str, context: str, config: dict) -> float:
    """
    Score how faithful the answer is to the context.
    Asks the LLM to judge on a scale of 0.0 to 1.0.
    """
    llm = ChatOpenAI(
        model=config["llm"]["model"],
        temperature=0
    )
    prompt = ChatPromptTemplate.from_template(
        "You are an evaluation assistant.\n\n"
        "Given the following context and answer, score how faithful "
        "the answer is to the context on a scale from 0.0 to 1.0.\n"
        "0.0 = answer is completely made up, not supported by context.\n"
        "1.0 = answer is fully supported by the context.\n\n"
        "Context:\n{context}\n\n"
        "Answer:\n{answer}\n\n"
        "Return ONLY a single decimal number between 0.0 and 1.0. "
        "No explanation."
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context[:2000], "answer": answer})

    try:
        return round(float(result.strip()), 4)
    except ValueError:
        return 0.0


def score_completeness(question: str, answer: str, config: dict) -> float:
    """
    Score how completely the answer addresses the question.
    Asks the LLM to judge on a scale of 0.0 to 1.0.
    """
    llm = ChatOpenAI(
        model=config["llm"]["model"],
        temperature=0
    )
    prompt = ChatPromptTemplate.from_template(
        "You are an evaluation assistant.\n\n"
        "Given the following question and answer, score how completely "
        "the answer addresses the question on a scale from 0.0 to 1.0.\n"
        "0.0 = answer does not address the question at all.\n"
        "1.0 = answer fully and completely addresses the question.\n\n"
        "Question:\n{question}\n\n"
        "Answer:\n{answer}\n\n"
        "Return ONLY a single decimal number between 0.0 and 1.0. "
        "No explanation."
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "answer": answer})

    try:
        return round(float(result.strip()), 4)
    except ValueError:
        0.0


def evaluate_strategy(
    strategy_name: str,
    retriever,
    config: dict
) -> list:
    """
    Run all benchmark questions against a strategy
    and return a list of result rows.
    """
    results = []
    questions = config["benchmark"]["questions"]

    print(f"\n  Evaluating: {strategy_name}")
    for i, question in enumerate(questions):
        print(f"    Question {i+1}/{len(questions)}: {question[:60]}...")

        answer, context, _ = get_answer(retriever, question, config)
        relevance = score_relevance(question, context, config)
        faithfulness = score_faithfulness(answer, context, config)
        completeness = score_completeness(question, answer, config)
        overall = round(
            (relevance + faithfulness + completeness) / 3, 4
        )

        results.append({
            "strategy": strategy_name,
            "question": question,
            "answer": answer,
            "relevance_score": relevance,
            "faithfulness_score": faithfulness,
            "completeness_score": completeness,
            "overall_score": overall
        })

    return results