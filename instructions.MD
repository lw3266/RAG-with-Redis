To start Redis server

> docker run -d --name redis-stack \
>   -p 6379:6379 -p 8001:8001 \
>   -v /INPUT/A/PATH/FOR/REDIS/TO/STORE/ITS/DATA:/data \
>   -v redis-stack.conf:/redis-stack.conf \
>   redis/redis-stack:latest \
>   redis-stack-server /redis-stack.conf

To stop Redis server
> docker stop redis-stack

To clear all content in the Redis database
> docker exec redis-stack redis-cli FLUSHDB

To start Redis
> docker restart redis-stack

First run ingest to write embedded data into Redis
> python3 ingest_embed_redis.py /PATH/TO/TRAINING/MATERIALS

Then retrieval-augmented generation in the command line with a language model
> python3 rag_cli_llm.py --host localhost --port 6379 --index doc_index

