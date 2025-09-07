# connect_memory_with_llm.py
import os
from typing import List

from langchain_huggingface import (
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
    ChatHuggingFace,   # <-- chat wrapper
)
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS


# ---------------------------
# Config
# ---------------------------
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
# Keep your model; your provider exposes it as *conversational* (chat), not text-generation
REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vector_store/db_faiss"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------
# Build Chat LLM (conversational)
# ---------------------------
def load_chat_model() -> ChatHuggingFace:
    if not HF_TOKEN:
        raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN in your environment.")
    # IMPORTANT: task="conversational"
    endpoint = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        task="conversational",
        temperature=0.5,
        max_new_tokens=512,
    )
    # Wrap endpoint as a Chat model so LangChain won’t call text_generation
    return ChatHuggingFace(llm=endpoint)


# ---------------------------
# Prompt
# ---------------------------
CUSTOM_PROMPT = """
Use ONLY the information in the context to answer the user's question.
If the answer is not in the context, say you don't know. Do not make anything up.

Context:
{context}

Question:
{question}

Answer:
""".strip()

def make_prompt(template: str) -> PromptTemplate:
    return PromptTemplate(template=template, input_variables=["context", "question"])


# ---------------------------
# Build Retrieval QA Chain
# ---------------------------
def build_chain() -> RetrievalQA:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    chat_model = load_chat_model()

    return RetrievalQA.from_chain_type(
        llm=chat_model,                                # <-- chat model here
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": make_prompt(CUSTOM_PROMPT)},
    )


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    qa = build_chain()

    query = input("Enter your query: ").strip()
    if not query:
        raise SystemExit("Empty query. Exiting.")

    # Newer retrievers prefer .invoke() over .get_relevant_documents()
    docs = qa.retriever.invoke(query)
    if not docs:
        print("(No matching context found in your PDFs — answering directly.)")
        # call the chat model directly when retrieval is empty
        chat = load_chat_model()
        print("\nAnswer:\n", chat.invoke(query).content)
    else:
        resp = qa.invoke({"query": query})
        print("\nAnswer:\n", resp["result"])
        print("\nSources:")
        for i, d in enumerate(resp.get("source_documents", []), 1):
            meta = d.metadata or {}
            print(f"[{i}] {meta.get('source','unknown')} (page {meta.get('page','n/a')})")
