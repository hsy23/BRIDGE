import json
from pathlib import Path
import random
import sys

import torch

from BRIDGE.utils.meter import TimeMeter, Statistics
from BRIDGE import generator
from BRIDGE import transceiver
from BRIDGE import decoder
from BRIDGE import bridge
from BRIDGE import aggregator
from BRIDGE.config import BRIDGEConfig
from BRIDGE.utils.configure import Field as F
from BRIDGE.utils.mlogging import Logger
from experiments.evaluator import Evaluator
from tqdm import tqdm
from BRIDGE.utils.stable import seed_everything

sys.path.append(".")


generator.logging_level = "DEBUG"
transceiver.logging_level = "DEBUG"
decoder.logging_level = "DEBUG"
bridge.logging_level = "DEBUG"
aggregator.logging_level = "DEBUG"
time_meter = TimeMeter()

LOGGER = Logger.build(__name__, level="INFO")

SYSTEM_PROMPT = (
    "You are now a helpful personal AI assistant. You should emulate the author's style "
    "and tone based on provided chat history content. Your responses should be detailed and "
    "informative, using the personal information reasonably in the user's profile. "
    "Aim for insightful and high-quality solutions that make users satisfied."
)

DEVICE_USER_TEMPLATE = """## User Profile
{profile}

## Chat History
{history}

## Task
{task}

## Instruction
Use the User Profile and Chat History to infer the user’s preferences, style, and tone, and complete the Task accordingly. The output should be natural, coherent, and context-aware. Do not include any self-introduction, persona framing, or first-person language. Do not restate or explain the task, and do not mention the template, instructions, or reasoning process. Keep the response concise yet sufficiently informative. Avoid numbered lists, bullet points, citations, or any meta-commentary.

## Response
"""

CLOUD_TEMPLATE = """## Additional Information
{context}

## Task
{query}

## Instruction
The output should be natural, coherent, and context-aware. Do not include any self-introduction, persona framing, or first-person language. Do not restate or explain the task, and do not mention the template, instructions, or reasoning process. Keep the response concise yet sufficiently informative. Avoid numbered lists, bullet points, citations, or any meta-commentary.

## Response
"""


class CoGenConfig(BRIDGEConfig):
    class evaluator:
        output_dir = F(str, default="outputs/cogen_", help="Output directory")
        data_path = F(str, default="/workspace/LSRP/data/CoGen/dev.json", help="CoGen dev.json path")
        n_prompts = F(int, default=10, help="Number of prompts per setting")
        max_new_tokens = F(int, default=256, help="Tokens per response")
        latencies = F(str, default="0,50,100,150,200,250,300", help="Comma-separated latency ms values")
        prompt_template = F(str, default=CLOUD_TEMPLATE, help="Cloud prompt")
        seed = F(int, default=42, help="Random seed")


def load_cogen(path: str, limit: int):
    with open(path, "r") as f:
        data = json.load(f)
    return random.sample(data, limit) if limit > 0 else data


def encode_texts(model, tokenizer, texts, device, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer.batch_encode_plus(batch, return_tensors="pt", padding="longest")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            emb = model(**enc)
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0)


def build_device_prompt(sample):
    return SYSTEM_PROMPT + "\n\n" + DEVICE_USER_TEMPLATE.format(
        profile=sample.get("additional_profile", ""),
        history="\n\n".join([profile.get("title", "") + "\n" + profile.get("text", "") for profile in sample.get("profile", [])]),
        task=sample["conversations"][0]["value"],
    )


class CoGenEvaluator(Evaluator):
    def __init__(self, config: CoGenConfig):
        super().__init__(config, name="CoGen")
        self.full_config = config
        self.dataset = load_cogen(config.evaluator.data_path, config.evaluator.n_prompts)
        self.device = bridge.BRIDGE(config)
        self.stats = Statistics()

    def _clear_stats(self):
        self.device.stats.records.clear()

    def evaluate(self):
        latencies = [int(x) for x in self.full_config.evaluator.latencies.split(",")]
        output_dir = Path(self.full_config.evaluator.output_dir, f"{self.run_id}-{self.full_config.aggregator.mode}")
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for latency in latencies:
            transceiver.LATENCY = latency
            transceiver.JITTER = latency / 5
            LOGGER.info(f"Setting latency to {latency} ms")

            outputs = []
            self._clear_stats()
            for sample in tqdm(self.dataset, desc="Processing prompts"):
                task = sample["conversations"][0]["value"]
                device_prompt = build_device_prompt(sample)
                with time_meter.timer("TotalGenerationTime") as t_timer:
                    output_text = self.device.query(
                        task,
                        self.full_config.evaluator.prompt_template,
                        self.full_config.evaluator.max_new_tokens,
                        local_context=device_prompt,
                    )
                run_time = t_timer.duration
                outputs.append(
                    {
                        "tid": sample.get("tid"),
                        "task": task,
                        "profile": sample.get("additional_profile", ""),
                        "chat_history": sample.get("profile", []),
                        "output": output_text,
                    }
                )

            result = {
                "latency_ms": latency,
                "n_docs": self.full_config.retriever.n_docs,
                "generate_time": run_time,
                "outputs": outputs,
            }
            results.append(result)

            stats_path = output_dir / f"stats_latency{latency}.json"
            from BRIDGE.aggregator import stats as aggregator_stats
            (self.device.stats | aggregator_stats).dump(stats_path)

            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)

        self.device.shutdown()


if __name__ == "__main__":
    config = CoGenConfig()
    config.parse_sys_args()
    seed_everything(config.evaluator.seed)
    config.generator.s_sequence = 1024
    config.retriever.s_context = 256
    config.trans.tx_port = 7010
    config.trans.rx_port = 7011

    if config.trans.rank == 0:
        config.trans.tx_host = "127.0.0.1"
        config.trans.tx_port, config.trans.rx_port = config.trans.rx_port, config.trans.tx_port
        bridge.BRIDGE(config)
    else:
        config.trans.tx_host = "127.0.0.1"
        evaluator = CoGenEvaluator(config)
        evaluator.evaluate()
