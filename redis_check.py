import redis
import binascii
import numpy

r = redis.Redis(host="localhost", port=6379)   # adjust if needed
DB_ID   = r.connection_pool.connection_kwargs.get("db", 0)
print(f"🔎 Connected to Redis DB {DB_ID}")

# # 1️⃣ Count keys
# key_count = r.dbsize()
# print(f"Total keys in DB: {key_count}")

# # 2️⃣ List first N keys (avoid KEYS * in prod)
# N = 2
# keys = [k for i, k in enumerate(r.scan_iter()) if i < N]

# print(f"First {len(keys)} keys ➜")
# for k in keys:
#     print("  ", k.decode())

# # 3️⃣ Inspect each key’s type + sample contents
# def pretty(val, maxlen=60):
#     """Readable preview of binary data."""
#     if isinstance(val, bytes):
#         hexed = binascii.hexlify(val[:24]).decode()
#         suffix = "…" if len(val) > 24 else ""
#         return f"<{len(val)}-byte blob {hexed}{suffix}>"
#     return val

# for k in keys:
#     t = r.type(k).decode()
#     print(f"\n🔸 Key: {k.decode()}  (type={t})")
#     if t == "hash":
#         data = r.hgetall(k)
#         for f, v in data.items():
#             print(f"   {f.decode():>10}: {pretty(v)}")
#     elif t == "string":
#         print("   value:", pretty(r.get(k)))
#     elif t in ("list", "set", "zset"):
#         print("   length:", r.llen(k) if t == "list" else r.scard(k))
#     else:
#         print("   (skipping detailed dump)")


