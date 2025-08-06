import argparse
import redis
from redis.commands.search.query import Query
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
)
from sentence_transformers import SentenceTransformer


# ----------------------------- CONFIG --------------------------------------

DOC_INDEX         = "doc_index_hybrid"
VECTOR_FIELD      = "embedding"
TOP_K             = 5
CONTEXT_TRUNCATE  = 3500
EMBED_MODEL_NAME   = "Qwen/Qwen3-Embedding-0.6B"
DEEPSEEK_CKPT     = "Deepseek-AI/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------- LOAD EMBEDDING MODEL --------------------------------

print(f"Loading text encoder ({EMBED_MODEL_NAME}) …")
device = "cuda" if torch.cuda.is_available() else "cpu"
text_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    trust_remote_code=True,
    device=device,
)

@torch.no_grad()
def embed(text: str) -> bytes:
    text_feats = text_model.encode(text, convert_to_tensor=True)

    # Normalize the vector (important for consistency with Redis vectors)
    text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)

    # Convert the normalized vector to a numpy array and serialize it to float32 bytes
    return text_feats.cpu().numpy().astype("float32").tobytes()

# -------------------- CONNECT TO REDIS & RETRIEVE --------------------------

def retrieve_chunks(r: redis.Redis, vec_blob: bytes, k: int = TOP_K):
    q = (
        Query(f"*=>[KNN {k} @{VECTOR_FIELD} $vec AS score]")
        .sort_by("score")
        .return_fields("doc_id", "type", "path", "page", "content")
        .dialect(2)
    )
    res = r.ft(DOC_INDEX).search(q, query_params={"vec": vec_blob})
    return res.docs

# ------------------- LOAD DEEPSEEK LLM -------------------------------------

print("Loading DeepSeek model …")
ds_tok = AutoTokenizer.from_pretrained(DEEPSEEK_CKPT, trust_remote_code=True)
ds_lm = AutoModelForCausalLM.from_pretrained(
    DEEPSEEK_CKPT,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
streamer = TextStreamer(ds_tok, skip_prompt=True, skip_special_tokens=True)

def generate_answer(question: str, source: str) -> str:
    system_prompt = (
        "Answer strictly based on the provided information. If the information is insufficient or irrelevant, say that you are unable to answer and end the chat."
        f"Here is the information:\n{source}\n\n"
    )
    full_prompt = system_prompt + f"Question: {question}\nAnswer:"
    inputs = ds_tok(full_prompt, return_tensors="pt").to(DEVICE)
    outputs = ds_lm.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=False,
        streamer=streamer,
    )
    text = ds_tok.decode(outputs[0], skip_special_tokens=True)
    # Cut off prompt portion
    return text.split("Answer:")[-1].strip()

# ---------------------------- CLI LOOP -------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--password", default=None)
    parser.add_argument("--index", default=DOC_INDEX)
    args = parser.parse_args()

    r = redis.Redis(host=args.host, port=args.port, password=args.password)
    print(f"Connected to Redis index: {args.index}")

    try:
        while True:
            question = input("\n💬 > ").strip()
            if not question:
                break

            print("\n📜 Generating answer…")
            vec_blob = embed(question)
            print(vec_blob)
            chunks = retrieve_chunks(r, vec_blob)

            if not chunks:
                print("No relevant context found.")
                continue

            source = "\n".join(f"[TEXT] {c.content}" for c in chunks)
            answer = generate_answer(question, source)
            print("\n📜 Answer:\n" + answer)

    except (KeyboardInterrupt, EOFError):
        print("\nBye!")

if __name__ == "__main__":
    main()
