import io, json, sys
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image
import pdfplumber, fitz
from tqdm import tqdm
import redis
import torch
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration

# --------------------------------------------------------------------- Config
REDIS_HOST, REDIS_PORT = "localhost", 6379

INDEX_NAME   = "doc_index_hybrid"
EMBED_DIM    = 1024
CHUNK_SIZE   = 2048
EMBED_FIELD  = "embedding"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- Embeddings
text_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    trust_remote_code=True,
    device=device,
)

processor   = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
captioner   = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(device).eval()


@torch.no_grad()
def embed_text(text: str) -> np.ndarray:
    return text_model.encode(text, normalize_embeddings=True)

@torch.no_grad()
def caption_image(pil: Image.Image) -> str:
    pil = pil.convert("RGB")  # Ensure 3-channel
    if pil.mode != "RGB":
        raise ValueError(f"Image mode is {pil.mode}, expected RGB")

    # Optional: check shape
    # arr = np.array(pil)
    # if arr.ndim != 3 or arr.shape[2] != 3:
    #     raise ValueError(f"Image shape is {arr.shape}, expected (H, W, 3)")

    inputs = processor(images=pil, return_tensors="pt", input_data_format="channels_last").to(device)
    out = captioner.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)



def embed_image(pil: Image.Image) -> np.ndarray:
    pil = pil.convert("RGB")
    caption = caption_image(pil)
    return embed_text(caption), caption

# ------------------------------------------------------------------- Redis
def create_index(r: redis.Redis) -> None:
    from redis.commands.search.field import TextField, VectorField
    from redis.commands.search.index_definition import IndexDefinition, IndexType

    try:
        r.ft(INDEX_NAME).info()
        return
    except redis.ResponseError:
        pass

    schema = (
        TextField("doc_id"),
        TextField("type"),
        TextField("path"),
        TextField("page"),
        TextField("content"),
        VectorField(
            EMBED_FIELD,
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": EMBED_DIM,
                "DISTANCE_METRIC": "COSINE",
                "INITIAL_CAP": 10000
            },
        ),
    )
    definition = IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH)
    r.ft(INDEX_NAME).create_index(schema, definition=definition)
    print(f"✅ Index {INDEX_NAME} created (dim={EMBED_DIM})")

def add_doc(r: redis.Redis, key: str, vec: np.ndarray, meta: Dict) -> None:
    r.hset(
        key,
        mapping={
            EMBED_FIELD: vec.astype("float32").tobytes(),
            **{k: str(v) for k, v in meta.items()},
        },
    )

# -------------------------------------------------------------- Extractors
def chunk_text(txt: str) -> List[str]:
    return [txt[i : i + CHUNK_SIZE] for i in range(0, len(txt), CHUNK_SIZE)]

def handle_txt(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    return [{"body": chunk, "type": "text", "page": 0} for chunk in chunk_text(raw)]

def handle_pdf(path: Path) -> List[Dict]:
    items = []
    with pdfplumber.open(path) as pdf:
        for pid, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            if txt.strip():
                items.extend(
                    {"body": chunk, "type": "text", "page": pid}
                    for chunk in chunk_text(txt)
                )
    doc = fitz.open(path)
    for page_index in range(len(doc)):
        for img in doc.get_page_images(page_index):
            xref = img[0]
            base = doc.extract_image(xref)
            print("image here: ", page_index)
            items.append(
                {
                    "body": base["image"],  # raw bytes
                    "type": "image",
                    "page": page_index + 1,
                }
            )
    return items

def handle_json(path: Path) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"⚠️ JSON parse failed: {e}")
            return []

    def flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}.{k}" if prefix else k
                yield from flatten(v, new_key)
        elif isinstance(obj, list):
            for idx, itm in enumerate(obj):
                yield from flatten(itm, f"{prefix}[{idx}]")
        elif isinstance(obj, str):
            yield prefix, obj

    for key, text in flatten(data):
        for chunk in chunk_text(text):
            items.append({"body": chunk, "type": "text", "page": 0, "section": key})
    return items

def handle_image(path: Path) -> List[Dict]:
    raw = Image.open(path).convert("RGB")
    return [{"body": raw, "type": "image", "page": 0}]

HANDLERS = {
    ".txt": handle_txt,
    ".md": handle_txt,
    ".pdf": handle_pdf,
    ".json": handle_json,
    ".png": handle_image,
    ".jpg": handle_image,
    ".jpeg": handle_image,
    ".webp": handle_image,
}

# -------------------------------------------------------------- Ingestion
def ingest_directory(root: str):
    r = redis.Redis(REDIS_HOST, REDIS_PORT, decode_responses=False)
    create_index(r)

    uid = 0
    for path in tqdm(list(Path(root).rglob("*.*"))):
        ext = path.suffix.lower()
        if ext not in HANDLERS:
            continue
        try:
            items = HANDLERS[ext](path)
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")
            continue

        for item in items:
            if item["type"] == "text":
                vec = embed_text(item["body"])
                content = item["body"][:120]
            else:
                if isinstance(item["body"], bytes):
                    pil = Image.open(io.BytesIO(item["body"])).convert("RGB")
                else:
                    pil = item["body"]
                vec, caption = embed_image(pil)
                content = caption

            key = f"doc:{uid}"
            meta = {
                "doc_id": path.stem,
                "type": item["type"],
                "path": str(path),
                "page": item["page"],
                "content": content,
            }
            add_doc(r, key, vec, meta)
            uid += 1

    print(f"🍰 Ingested {uid} chunks (text+captions) into Redis.")

# --------------------------------------------------------------- __main__
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest_multimodal_redis.py <folder>")
        sys.exit(1)
    ingest_directory(sys.argv[1])
