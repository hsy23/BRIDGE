import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

from BRIDGE import aggregator
from BRIDGE import bridge
from BRIDGE import decoder
from BRIDGE import generator
from BRIDGE import transceiver
from BRIDGE.config import BRIDGEConfig
from BRIDGE.utils.configure import Field as F
from BRIDGE.utils.meter import TimeMeter
from BRIDGE.utils.mlogging import Logger
from BRIDGE.utils.stable import seed_everything
from experiments.NaturalQuestions.calc_metrics import score_predictions, write_tsv_artifacts
from experiments.evaluator import Evaluator

sys.path.append(".")


generator.logging_level = "DEBUG"
transceiver.logging_level = "DEBUG"
decoder.logging_level = "DEBUG"
bridge.logging_level = "DEBUG"
aggregator.logging_level = "DEBUG"
time_meter = TimeMeter()

LOGGER = Logger.build(__name__, level="INFO")
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "datasets" / "NaturalQuestions" / "dev.jsonl"

PROMPT_TEMPLATE = """Read the evidence and answer the question with the shortest exact answer phrase.
Evidence: {context}
Question: {query}
Short answer:"""


def normalize_generation(text: str) -> str:
    text = text.strip()
    text = text.splitlines()[0] if text else ""
    text = text.split("<")[0]
    return text.strip()


class NaturalQuestionsConfig(BRIDGEConfig):
    class retriever(BRIDGEConfig.retriever):
        passages = F(str, default="wikipedia[remote]", help="Passage source for open-domain QA")
        n_docs = F(int, default=4, help="Number of retrieved documents")
        s_aggregate = F(int, default=4, help="Number of retrieved passages to aggregate")
        s_context = F(int, default=128, help="Maximum number of tokens allowed for passage contexts")

    class trans(BRIDGEConfig.trans):
        tx_port = F(int, default=7020, help="Port for sending data")
        rx_port = F(int, default=7021, help="Port for receiving data")

    class generator(BRIDGEConfig.generator):
        s_sequence = F(int, default=896, help="Maximum input sequence length")
        use_fp16 = F(bool, default=True, help="Use fp16 for generation")

    class sampler(BRIDGEConfig.sampler):
        do_sample = F(bool, default=False, help="Whether to sample during decoding")

    class evaluator:
        output_dir = F(str, default="outputs/natural_questions_", help="Output directory")
        data_path = F(str, default=str(DEFAULT_DATA_PATH), help="NaturalQuestions dev dataset path")
        n_samples = F(int, default=20, help="Number of evaluation examples")
        max_new_tokens = F(int, default=10, help="Maximum number of generated tokens")
        latencies = F(str, default="0", help="Comma-separated latency ms values")
        prompt_template = F(str, default=PROMPT_TEMPLATE, help="Prompt template")
        seed = F(int, default=42, help="Random seed")
        random_sample = F(bool, default=False, help="Whether to sample examples randomly")


def load_natural_questions(path: str, limit: int, seed: int, random_sample: bool) -> list[dict]:
    dataset_path = Path(path)
    if dataset_path.suffix == ".jsonl":
        data = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))

    normalized = []
    for idx, item in enumerate(data):
        answers = item.get("answers", item.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]
        normalized.append(
            {
                "id": item.get("id", f"nq_{idx + 1:04d}"),
                "question": item["question"],
                "answers": answers,
            }
        )

    data = normalized
    if random_sample and limit > 0 and limit < len(data):
        rng = random.Random(seed)
        return rng.sample(data, limit)
    if limit > 0:
        return data[:limit]
    return data


class NaturalQuestionsEvaluator(Evaluator):
    def __init__(self, config: NaturalQuestionsConfig):
        super().__init__(config, name="NaturalQuestions")
        self.full_config = config
        self.dataset = load_natural_questions(
            config.evaluator.data_path,
            config.evaluator.n_samples,
            config.evaluator.seed,
            config.evaluator.random_sample,
        )
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

            predictions = []
            outputs = []
            self._clear_stats()
            for sample in tqdm(self.dataset, desc="Processing questions"):
                question = sample["question"]
                with time_meter.timer("TotalGenerationTime") as t_timer:
                    output_text = self.device.query(
                        question,
                        self.full_config.evaluator.prompt_template,
                        self.full_config.evaluator.max_new_tokens,
                    )
                run_time = t_timer.duration
                prediction = normalize_generation(output_text)
                predictions.append(prediction)
                outputs.append(
                    {
                        "id": sample.get("id"),
                        "question": question,
                        "answers": sample["answers"],
                        "prediction": prediction,
                        "raw_output": output_text,
                    }
                )

            f1, em, per_item = score_predictions(predictions, self.dataset)
            pred_path = output_dir / "qa_preds.tsv"
            gold_path = output_dir / "dev.question_answers"
            write_tsv_artifacts(pred_path, gold_path, predictions, self.dataset)

            merged_stats = self.device.stats | aggregator.stats
            merged_stats.dump(output_dir / f"stats_latency{latency}.json")
            transceiver.stats.dump(output_dir / f"transceiver_stats_latency{latency}.json")

            metric_payload = {"f1": f1, "em": em}
            (output_dir / "metric_score.txt").write_text(
                f"f1: {metric_payload['f1']}\nem: {metric_payload['em']}\n",
                encoding="utf-8",
            )

            result = {
                "latency_ms": latency,
                "n_docs": self.full_config.retriever.n_docs,
                "generate_time": run_time,
                "metrics": metric_payload,
                "outputs": outputs,
            }
            results.append(result)

            with open(output_dir / "intermediate_data.json", "w", encoding="utf-8") as f:
                json.dump(per_item, f, indent=2, ensure_ascii=False)
            with open(output_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        self.device.shutdown()


if __name__ == "__main__":
    config = NaturalQuestionsConfig()
    config.parse_sys_args()
    seed_everything(config.evaluator.seed)

    if config.trans.rank == 0:
        config.trans.tx_host = "127.0.0.1"
        config.trans.tx_port, config.trans.rx_port = config.trans.rx_port, config.trans.tx_port
        bridge.BRIDGE(config)
    else:
        config.trans.tx_host = "127.0.0.1"
        evaluator = NaturalQuestionsEvaluator(config)
        evaluator.evaluate()
