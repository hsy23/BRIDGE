.PHONY: retriever_server

RETRIEVER_PORT ?= 8765

retriever_server:
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python dragon/toolbox/retriever_as_a_service.py $(RETRIEVER_PORT)

cogen_pareto_cloud:
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python -u experiments/CoGen/eval.py \
	--trans.rank 0 \
	--aggregator.mode BRIDGE \
	--device cuda:0 \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--retriever.passages "wikipedia[remote]" \
	--retriever.n_docs 8 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765 > cloud.log 2>cloud_err.log

cogen_pareto_device:
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python -u experiments/CoGen/eval.py \
	--trans.rank 1 \
	--retriever.n_docs 0 \
	--aggregator.mode BRIDGE \
	--device cpu \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--evaluator.n_prompts 1 \
	--evaluator.latencies 0 \
	--evaluator.max_new_tokens 20 \
	--retriever.passages "cogen" \
	--retriever.s_aggregate 1 > device.log 2>device_err.log

cogen:
	$(MAKE) cogen_pareto_cloud & \
	$(MAKE) cogen_pareto_device &

cogen_crcg_cloud:
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python -u experiments/CoGen/eval_crcg.py \
	--generator.model "models/Qwen/Qwen2.5-7B" \
	--generator.use_fp16 \
	--retriever.passages "wikipedia[remote]" \
	--retriever.n_docs 0 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765 \
	--evaluator.n_prompts 30 \
	--evaluator.max_new_tokens 256 \
	--evaluator.wiki

cogen_crcg_device_cogen:
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python -u experiments/CoGen/eval_crcg.py \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--retriever.passages "cogen" \
	--retriever.n_docs 0 \
	--evaluator.n_prompts 30 \
	--evaluator.max_new_tokens 256 \
	--evaluator.cogen

cogen_crcg_device_cogen_wiki:
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python -u experiments/CoGen/eval_crcg.py \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--retriever.passages "wikipedia[remote]" \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765 \
	--evaluator.n_prompts 30 \
	--evaluator.max_new_tokens 256 \
	--evaluator.wiki \
	--evaluator.cogen

cogen_crcg_cloud_cogen_wiki:
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python -u experiments/CoGen/eval_crcg.py \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--retriever.passages "wikipedia[remote]" \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765 \
	--evaluator.n_prompts 30 \
	--evaluator.max_new_tokens 256 \
	--evaluator.wiki \
	--evaluator.cogen

cogen_score:
	PYTHONPATH=/workspace/BRIDGE python -u experiments/CoGen/score.py \
	--results_path "outputs/cogen_/CoGen-20260206120606-Qwen2.5-7B-wiki/results.json" \
	--output_path "outputs/cogen_/CoGen-20260206120606-Qwen2.5-7B-wiki/scored.json" \
	--api_base "https://api.996444.cn/v1/" \
	--api_key "sk-0uXyQElaBxwWl2uZc1HNhdIAvkRQlYmMmy8kA58MTgoe9eA5"






movieexp_cloud:
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python -u experiments/MovieExp/eval.py \
	--trans.rank 0 \
	--aggregator.mode BRIDGE \
	--device cuda:0 \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--retriever.passages "wikipedia[remote]" \
	--retriever.n_docs 8 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765 > cloud.log 2>cloud_err.log

movieexp_device:
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python -u experiments/MovieExp/eval.py \
	--trans.rank 1 \
	--retriever.n_docs 0 \
	--aggregator.mode BRIDGE \
	--device cpu \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--evaluator.n_prompts 30 \
	--evaluator.latencies 0 \
	--evaluator.max_new_tokens 256 \
	--retriever.passages "movieexp" \
	--retriever.s_aggregate 1 > device.log 2>device_err.log

movieexp:
	$(MAKE) movieexp_cloud & \
	$(MAKE) movieexp_device &

movieexp_cloud_dragon:
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python -u experiments/MovieExp/eval.py \
	--trans.rank 0 \
	--aggregator.mode speculative \
	--device cuda:0 \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--retriever.passages "wikipedia[remote]" \
	--retriever.n_docs 8 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765 > cloud.log 2>cloud_err.log

movieexp_device_dragon:
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python -u experiments/MovieExp/eval.py \
	--trans.rank 1 \
	--retriever.n_docs 0 \
	--aggregator.mode speculative \
	--device cpu \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--evaluator.n_prompts 30 \
	--evaluator.latencies 0 \
	--evaluator.max_new_tokens 256 \
	--retriever.passages "movieexp" \
	--retriever.s_aggregate 1 > device.log 2>device_err.log

movieexp_dragon:
	$(MAKE) movieexp_cloud_dragon & \
	$(MAKE) movieexp_device_dragon &

movieexp_score:
	PYTHONPATH=/workspace/BRIDGE python -u experiments/MovieExp/score.py \
	--results_path "outputs/cogen_/CoGen-20260206120606-Qwen2.5-7B-wiki/results.json" \
	--output_path "outputs/cogen_/CoGen-20260206120606-Qwen2.5-7B-wiki/scored.json" \
	--api_base "https://api.996444.cn/v1/" \
	--api_key "sk-0uXyQElaBxwWl2uZc1HNhdIAvkRQlYmMmy8kA58MTgoe9eA5"

movieexp_cloud_crcg:
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python -u experiments/MovieExp/eval_crcg.py \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--retriever.passages "wikipedia[remote]" \
	--retriever.n_docs 0 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765 \
	--evaluator.n_prompts 30 \
	--evaluator.max_new_tokens 256 \
	--evaluator.wiki

movieexp_device_crcg:
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python -u experiments/MovieExp/eval_crcg.py \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--retriever.passages "movieexp" \
	--retriever.n_docs 0 \
	--evaluator.n_prompts 30 \
	--evaluator.max_new_tokens 256 \
	--evaluator.movieexp

movieexp_device_movieexp_wiki:
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python -u experiments/MovieExp/eval_crcg.py \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--retriever.passages "wikipedia[remote]" \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765 \
	--evaluator.n_prompts 30 \
	--evaluator.max_new_tokens 256 \
	--evaluator.wiki \
	--evaluator.movieexp

movieexp_cloud_movieexp_wiki:
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python -u experiments/MovieExp/eval_crcg.py \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--retriever.passages "wikipedia[remote]" \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765 \
	--evaluator.n_prompts 30 \
	--evaluator.max_new_tokens 256 \
	--evaluator.wiki \
	--evaluator.movieexp








sample_prompts:
	mkdir -p datasets/prompts
	HF_HOME=/workspace/.cache/huggingface \
	PYTHONPATH=/workspace/BRIDGE python dragon/toolbox/sample_prompts.py

kill:
	pgrep -f Qwen | xargs kill -9 || echo "No Qwen process found"
kill_llama:
	pgrep -f llama | xargs kill -9 || echo "No llama process found"

latency_dragon_cloud:
	HF_HOME=/workspace/.cache/huggingface \
	python experiments/Latency/eval_no_shutdown.py \
	--trans.rank 0 \
	--aggregator.mode speculative \
	--device cuda:0 \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "models/llama3/Llama-3.1-8B-Instruct" \
	--generator.use_fp16 \
	--evaluator.n_prompts 3 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765  > cloud.log 2>cloud_err.log

latency_dragon_device:
	HF_HOME=/workspace/.cache/huggingface \
	echo "DRAGON" | python experiments/Latency/eval_no_shutdown.py \
	--trans.rank 1 \
	--device cpu \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "models/llama3/Llama-3.1-8B-Instruct" \
	--evaluator.max_new_tokens 256 \
	--generator.use_fp16 \
	--aggregator.mode speculative \
	--evaluator.n_prompts 1 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765 > device.log 2>device_err.log

latency_blockagg_cloud:
	HF_HOME=/workspace/.cache/huggingface \
	python -u experiments/Latency/eval_no_shutdown.py \
	--trans.rank 0 \
	--aggregator.mode BRIDGE \
	--device cuda:0 \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "models/llama3/Llama-3.1-8B-Instruct" \
	--generator.use_fp16 \
	--evaluator.n_prompts 1 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765 > cloud.log 2>cloud_err.log

latency_blockagg_device:
	HF_HOME=/workspace/.cache/huggingface \
	echo "BRIDGE" | python -u experiments/Latency/eval_no_shutdown.py \
	--trans.rank 1 \
	--device cpu \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "models/llama3/Llama-3.1-8B-Instruct" \
	--evaluator.max_new_tokens 256 \
	--generator.use_fp16 \
	--aggregator.mode BRIDGE \
	--evaluator.n_prompts 1 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765 > device.log 2>device_err.log

latency_sync_cloud:
	HF_HOME=/workspace/.cache/huggingface \
	python -u experiments/Latency/eval_no_shutdown.py \
	--trans.rank 0 \
	--aggregator.mode synchronized \
	--device cuda:0 \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "models/llama3/Llama-3.1-8B-Instruct" \
	--generator.use_fp16 \
	--evaluator.n_prompts 3 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765 > cloud.log 2>cloud_err.log

latency_sync_device:
	HF_HOME=/workspace/.cache/huggingface \
	echo "synchronized" | python -u experiments/Latency/eval_no_shutdown.py \
	--trans.rank 1 \
	--device cpu \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "models/llama3/Llama-3.1-8B-Instruct" \
	--evaluator.max_new_tokens 256 \
	--generator.use_fp16 \
	--aggregator.mode synchronized \
	--evaluator.n_prompts 1 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765 > device.log 2>device_err.log

latency_blockagg:
	$(MAKE) latency_blockagg_cloud & \
	sleep 1; \
	$(MAKE) latency_blockagg_device &

latency_dragon:
	$(MAKE) latency_dragon_cloud & \
	sleep 1; \
	$(MAKE) latency_dragon_device &

latency_sync:
	$(MAKE) latency_sync_cloud & \
	sleep 1; \
	$(MAKE) latency_sync_device &

latency_drdg_tw_cloud:
	python experiments/Latency/eval.py \
	--trans.rank 0 \
	--aggregator.mode synchronized \
	--device cuda:0 \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--evaluator.n_prompts 3 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765

latency_drdg_tw_device:
	python experiments/Latency/eval.py \
	--trans.rank 1 \
	--device cpu \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "Qwen/Qwen2.5-1.5B" \
	--evaluator.max_new_tokens 256 \
	--generator.use_fp16 \
	--aggregator.mode synchronized \
	--evaluator.n_prompts 3 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8767

latency_drdg_tw: latency_drdg_tw_cloud latency_drdg_tw_device

latency_drdg_sw_cloud:
	python experiments/Latency/eval_DRDG_SW.py \
	--trans.rank 0 \
	--device cuda:0 \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--evaluator.n_prompts 3 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765

latency_drdg_sw_device:
	python experiments/Latency/eval_DRDG_SW.py \
	--trans.rank 1 \
	--device cpu \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "Qwen/Qwen2.5-1.5B" \
	--evaluator.max_new_tokens 256 \
	--generator.use_fp16 \
	--evaluator.n_prompts 3 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8767

latency_drdg_sw: latency_drdg_sw_cloud latency_drdg_sw_device

latency_crcg_cloud:
	python experiments/Latency/eval_CRCG.py \
	--device cuda:0 \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--evaluator.max_new_tokens 256 \
	--generator.use_fp16 \
	--evaluator.n_prompts 1 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.downsample_type 2

latency_crcg_device:
	python experiments/Latency/eval_CRCG.py \
	--device cpu \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "models/Qwen/Qwen2.5-1.5B" \
	--evaluator.max_new_tokens 256 \
	--generator.use_fp16 \
	--evaluator.n_prompts 1 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.downsample_type 1

clean:
	rm -rf outputs/*
	rm -rf logs/*

# ==========================
# TTFT (Qwen) latency-style
# ==========================

.PHONY: latency_ttft_prefix latency_ttft_prefix_cloud latency_ttft_prefix_device \
		latency_ttft_retrieval latency_ttft_retrieval_cloud latency_ttft_retrieval_device \
		latency_ttft latency_ttft_cloud latency_ttft_device \
		latency_ttft_download_kv latency_ttft_download_kv_cloud latency_ttft_download_kv_device

ttft_prefix_proportion:
	python experiments/TTFT/eval_prefix_proportion.py \
	--trans.rank 0 \
	--device cpu \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--evaluator.n_prompts 3 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765

ttft_retrieval_cloud:
	python experiments/TTFT/eval_retrieval_latency.py \
	--trans.rank 0 \
	--device cuda:0 \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--evaluator.n_prompts 3 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765

ttft_retrieval_device:
	python experiments/TTFT/eval_retrieval_latency.py \
	--trans.rank 1 \
	--device cpu \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--evaluator.n_prompts 3 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8767

ttft_cloud:
	python experiments/TTFT/eval.py \
	--trans.rank 0 \
	--device cuda:0 \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--evaluator.n_prompts 3 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765

ttft_device:
	python experiments/TTFT/eval.py \
	--trans.rank 1 \
	--device cpu \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--evaluator.n_prompts 3 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8767

ttft:
	$(MAKE) ttft_cloud & \
	sleep 2; \
	$(MAKE) ttft_device &

ttft_download_kv_cloud:
	python experiments/TTFT/eval_download_kv.py \
	--trans.rank 0 \
	--device cuda:0 \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--evaluator.n_prompts 3 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8765

ttft_download_kv_device:
	python experiments/TTFT/eval_download_kv.py \
	--trans.rank 1 \
	--device cpu \
	--retriever.passages "wikipedia[remote]" \
	--generator.model "Qwen/Qwen2.5-1.5B" \
	--generator.use_fp16 \
	--evaluator.n_prompts 3 \
	--retriever.n_docs 2 \
	--retriever.s_aggregate 1 \
	--retriever.host 127.0.0.1 \
	--retriever.port 8767

ttft_download_kv:
	$(MAKE) ttft_download_kv_cloud & \
	sleep 2; \
	$(MAKE) ttft_download_kv_device &
