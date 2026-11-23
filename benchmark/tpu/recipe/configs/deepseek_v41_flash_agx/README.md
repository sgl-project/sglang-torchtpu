# DeepSeek-V4.1-Flash AgentX recipes

`deepseek_v41_flash_dspark_c<N>.py` is the AgentX serving recipe for concurrency N
(c4 … c256) on one tpu7x 2x2x1 host, attention DP8 / EP8 with DSpark. Each builds on
`deepseek_v41_flash_dspark_agentx.py` (AgentX sizing) and `deepseek_v41_flash_dspark.py`
(DSpark on `deepseek_v41_flash.py`), and overrides only what its concurrency needs.

    python3 -m benchmark.tpu.launch_server \
        --config recipe/configs/deepseek_v41_flash_agx/deepseek_v41_flash_dspark_c64.py

## Session affinity (`affinity_proxy.py`)

With attention DP every rank keeps its own radix cache, so an agentic session must
reach the same rank turn after turn to hit its prefix. `affinity_proxy.py` sits in
front of the server and pins each conversation (and its subagents) to one DP rank
through the `X-Data-Parallel-Rank` header, using the Dynamo session headers AIPerf
sends.

    # server on :30000 (one of the recipes above), proxy on :30001
    python3 benchmark/tpu/recipe/configs/deepseek_v41_flash_agx/affinity_proxy.py \
        --port 30001 --upstream-port 30000 --dp 8 \
        --log proxy.jsonl --summary proxy_summary.json

    # client: point AIPerf at the proxy and have it send the session headers
    AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true \
        aiperf profile --url http://localhost:30001 ...

`--dp` must match the server's attention DP size. Requests without a session header
are forwarded unpinned (the server's DP dispatcher places them).
