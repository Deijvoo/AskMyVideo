import argparse
from pathlib import Path
from langchain_community.llms import Ollama
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

def main():
    parser = argparse.ArgumentParser(description="Ask questions over a FAISS RAG index.")
    parser.add_argument("faiss_path", type=Path, help="Path to .faiss index")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument("--model", type=str, default="llama3", help="LLM model name (default: llama3)")
    args = parser.parse_args()

    # Try OllamaEmbeddings, fallback to SentenceTransformerEmbeddings
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    if not args.faiss_path.exists():
        print(f"FAISS index not found at {args.faiss_path}. Please run chunk_embed.py first.")
        return

    index = FAISS.load_local(str(args.faiss_path), embeddings, allow_dangerous_deserialization=True)
    retriever = index.as_retriever(search_kwargs={"k": 4, "return_source_documents": True})

    llm = Ollama(model=args.model)
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    print("Querying index...")
    result = qa.invoke({"query": args.question})
    answer = result.get("result", result)
    print("Answer:", answer)

    # Print citations (timestamps and text snippets)
    source_docs = result.get("source_documents")
    if source_docs:
        print("\nCitations:")
        for doc in source_docs:
            meta = doc.metadata
            start = meta.get("start", "?")
            end = meta.get("end", "?")
            snippet = doc.page_content[:120].replace("\n", " ") + ("..." if len(doc.page_content) > 120 else "")
            print(f"- {start} --> {end}: {snippet}")

if __name__ == "__main__":
    main() 