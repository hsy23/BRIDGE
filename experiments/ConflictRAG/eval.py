import json
from pathlib import Path
import random
import sys

from BRIDGE.utils.meter import TimeMeter
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
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "datasets" / "ConflictRAG" / "test.json"

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

CLOUD_TEMPLATE = """## Remote Notes
{context}

## Task
{query}

## Instruction
The output should be natural, coherent, and context-aware. Do not include any self-introduction, persona framing, or first-person language. Do not restate or explain the task, and do not mention the template, instructions, or reasoning process. Keep the response concise yet sufficiently informative. Avoid numbered lists, bullet points, citations, or any meta-commentary.

## Response
"""


class ConflictRAGConfig(BRIDGEConfig):
    class retriever(BRIDGEConfig.retriever):
        passages = F(str, default="unused", help="Unused for ConflictRAG because retrieval is disabled")
        n_docs = F(int, default=0, help="ConflictRAG uses inline contexts instead of retrieval")
        s_context = F(int, default=256, help="Maximum number of tokens allowed for local context")

    class generator(BRIDGEConfig.generator):
        s_sequence = F(int, default=1024, help="Maximum input sequence length")

    class trans(BRIDGEConfig.trans):
        tx_port = F(int, default=7010, help="Port for sending data")
        rx_port = F(int, default=7011, help="Port for receiving data")

    class evaluator:
        output_dir = F(str, default="outputs/conflictrag_", help="Output directory")
        data_path = F(str, default=str(DEFAULT_DATA_PATH), help="ConflictRAG test.json path")
        n_prompts = F(int, default=20, help="Number of prompts per setting")
        max_new_tokens = F(int, default=256, help="Tokens per response")
        latencies = F(str, default="0", help="Comma-separated latency ms values")
        prompt_template = F(str, default=CLOUD_TEMPLATE, help="Cloud prompt")
        seed = F(int, default=42, help="Random seed")


def load_conflict_rag(path: str, limit: int):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return random.sample(data, limit) if limit > 0 and limit < len(data) else data


def build_local_prompt(sample):
    return SYSTEM_PROMPT + "\n\n" + DEVICE_USER_TEMPLATE.format(
        profile=sample["local_context"],
        history="",
        task=sample["conversations"][0]["value"],
    )


def build_remote_prompt(sample):
    return SYSTEM_PROMPT + "\n\n" + CLOUD_TEMPLATE.format(
        context=sample["cloud_context"],
        query=sample["conversations"][0]["value"],
    )


class ConflictRAGEvaluator(Evaluator):
    def __init__(self, config: ConflictRAGConfig):
        super().__init__(config, name="ConflictRAG")
        self.full_config = config
        self.dataset = load_conflict_rag(config.evaluator.data_path, config.evaluator.n_prompts)
        self.device = bridge.BRIDGE(config)

    def _clear_stats(self):
        self.device.stats.clear()
        aggregator.stats.clear()
        decoder.stats.clear()
        transceiver.stats.clear()

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
                remote_prompt = build_remote_prompt(sample)
                local_prompt = build_local_prompt(sample)

                with time_meter.timer("TotalGenerationTime") as t_timer:
                    output_text = self.device.query(
                        remote_prompt,
                        self.full_config.evaluator.prompt_template,
                        self.full_config.evaluator.max_new_tokens,
                        local_context=local_prompt,
                    )
                run_time = t_timer.duration

                outputs.append(
                    {
                        "task": sample["conversations"][0]["value"],
                        "cloud_context": sample.get("cloud_context"),
                        "local_context": sample.get("local_context"),
                        "reference_answer": sample["conversations"][1]["value"] if len(sample.get("conversations", [])) > 1 else "",
                        "output": output_text,
                    }
                )

            merged_stats = self.device.stats | aggregator.stats
            stats_path = output_dir / f"stats_latency{latency}.json"
            merged_stats.dump(stats_path)

            result = {
                "latency_ms": latency,
                "n_docs": self.full_config.retriever.n_docs,
                "generate_time": run_time,
                "outputs": outputs,
            }
            results.append(result)

            with open(output_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        self.device.shutdown()


if __name__ == "__main__":
    config = ConflictRAGConfig()
    config.parse_sys_args()
    seed_everything(config.evaluator.seed)

    if config.trans.rank == 0:
        config.trans.tx_host = "127.0.0.1"
        config.trans.tx_port, config.trans.rx_port = config.trans.rx_port, config.trans.tx_port
        bridge.BRIDGE(config)
    else:
        config.trans.tx_host = "127.0.0.1"
        evaluator = ConflictRAGEvaluator(config)
        evaluator.evaluate()
