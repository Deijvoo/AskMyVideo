import argparse
from pathlib import Path
import requests
import base64
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import json
import cv2
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

DEFAULT_MODEL_VISION = "llava:7b-v1.6-mistral-q4_0"
DEFAULT_MODEL_TEXT = "llama3"


def preload_images(index):
    """Load, resize, and encode all unique images from the FAISS index."""
    print("Pre-loading images...")
    all_image_paths = set()
    for doc in index.docstore._dict.values():
        frames = doc.metadata.get("frames", [])
        for frame in frames:
            all_image_paths.add(frame)

    images_b64_cache = {}
    for img_path in all_image_paths:
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (320, 180))
            _, buffer = cv2.imencode(".jpg", img)
            img_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
            images_b64_cache[img_path] = img_b64
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
    print(f"Pre-loaded {len(images_b64_cache)} images.")
    return images_b64_cache


def ask_question_vision(question, retriever, model, image_cache, chat_history, args):
    """Ask a multimodal question to LLaVA."""
    print("-> Searching for relevant context in the video...")
    docs = retriever.invoke(question)

    # Collect all unique image paths from the relevant chunks
    image_paths = set()
    context_chunks = []
    for doc in docs:
        frames = doc.metadata.get("frames", [])
        for frame in frames:
            image_paths.add(frame)
        context_chunks.append(doc.page_content)
    image_paths = list(image_paths)

    images_b64 = []
    if not args.no_images:
        if not image_paths:
            print("No image context found – falling back to text-only answer.")
        else:
            # Retrieve pre-loaded images from cache
            images_b64 = [image_cache[path] for path in image_paths if path in image_cache]

    history_str = "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_history])
    context_str = "\n---\n".join(context_chunks)
    prompt = (
        "You are a robot assistant. Your only function is to answer questions based *only* on the text provided in the 'CONTEXT'.\n"
        "- Your answer MUST start with a direct quote from the CONTEXT.\n"
        "- If the CONTEXT does not contain the answer, you MUST reply with ONLY the words: 'Answer not found in context.'\n\n"
        "CONTEXT:\n"
        f"{context_str}\n\n"
        "CONVERSATION HISTORY:\n"
        f"{history_str}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "ANSWER:"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "images": images_b64,
        "stream": True,
    }

    print("-> Context and prompt prepared. Sending to LLaVA model...")
    print("-> Waiting for AI response...")
    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        stream=True,
        timeout=600,
    )
    if response.ok:
        print("\nLLaVA answer:")
        full_answer = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                full_answer += data.get("response", "")
        full_answer = full_answer.strip()
        print(full_answer)
        return full_answer
    else:
        print(f"Ollama API error: {response.status_code} {response.text}")
        return None


def ask_question_text(question, retriever, model):
    """Ask a text-only question to a local LLM."""
    print("-> Querying text model...")
    llm = Ollama(model=model)
    qa_chain = RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    result = qa_chain.invoke({"query": question})
    answer = result.get("result", "")
    
    print("\nAnswer:")
    print(answer.strip())

    source_docs = result.get("source_documents")
    if source_docs:
        print("\nCitations:")
        for doc in source_docs:
            start = doc.metadata.get("start")
            end = doc.metadata.get("end")
            snippet = doc.page_content.replace('\n', ' ')[:100] + "..."
            
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                print(f"- From {start:.2f}s to {end:.2f}s: \"{snippet}\"")
            else:
                start_str = start if start is not None else "??"
                end_str = end if end is not None else "??"
                print(f"- From {start_str} to {end_str}: \"{snippet}\"")


def main():
    parser = argparse.ArgumentParser(
        description="Ask a question over a FAISS RAG index with multimodal or text-only context."
    )
    parser.add_argument("faiss_path", type=Path, help="Path to .faiss index")
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default=None,
        help="Question to ask (optional; if not provided, will enter interactive mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_VISION,
        help=f"LLM model name (default: {DEFAULT_MODEL_VISION})",
    )
    parser.add_argument(
        "--k", type=int, default=4, help="Number of relevant chunks to retrieve"
    )
    parser.add_argument(
        "--no-images", action="store_true", help="Run in text-only mode (uses a different model)."
    )
    args = parser.parse_args()

    embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    if not args.faiss_path.exists():
        print(
            f"FAISS index not found at {args.faiss_path}. Please run chunk_embed.py first."
        )
        return

    print("Loading FAISS index...")
    index = FAISS.load_local(
        str(args.faiss_path), embeddings, allow_dangerous_deserialization=True
    )
    retriever = index.as_retriever(
        search_kwargs={"k": args.k, "return_source_documents": True}
    )
    
    image_cache = {}
    if not args.no_images:
        image_cache = preload_images(index)

    if args.no_images:
        model = args.model if args.model != DEFAULT_MODEL_VISION else DEFAULT_MODEL_TEXT
        print(f"Running in text-only mode with model: {model}")
        
        if args.question:
            # Handle single text question
            ask_question_text(args.question, retriever, model)
        else:
            # Interactive text chat
            print("Entering text-only interactive mode. Type 'exit' to quit.")
            while True:
                try:
                    question = input("You: ")
                    if question.lower() == "exit":
                        break
                    ask_question_text(question, retriever, model)
                except KeyboardInterrupt:
                    break
            print("\nExiting chat.")
    else:
        # Vision mode
        model = args.model if args.model != DEFAULT_MODEL_TEXT else DEFAULT_MODEL_VISION
        
        if args.question:
            ask_question_vision(args.question, retriever, model, image_cache, [], args)
        else:
            print("Entering interactive mode. Type 'exit' to quit.")
            chat_history = []
            while True:
                try:
                    question = input("You: ")
                    if question.lower() == "exit":
                        break
                    answer = ask_question_vision(
                        question, retriever, model, image_cache, chat_history, args
                    )
                    if answer:
                        chat_history.append((question, answer))
                        if len(chat_history) > 3:  # Keep last 3 exchanges
                            chat_history.pop(0)
                except KeyboardInterrupt:
                    break
            print("\nExiting chat.")


if __name__ == "__main__":
    main() 