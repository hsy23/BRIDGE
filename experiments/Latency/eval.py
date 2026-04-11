import sys
import json
import time
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

sys.path.append(".")

from BRIDGE import generator
from BRIDGE import transceiver
from BRIDGE import decoder
from BRIDGE import bridge
from BRIDGE import aggregator
logging_level = "DEBUG"
generator.logging_level = logging_level
transceiver.logging_level = logging_level
decoder.logging_level = logging_level
bridge.logging_level = logging_level
aggregator.logging_level = logging_level

from BRIDGE.config import BRIDGEConfig
from BRIDGE.bridge import BRIDGE
from BRIDGE.utils.stable import seed_everything
from BRIDGE.utils.configure import Field as F
from experiments.evaluator import Evaluator
from BRIDGE.transceiver import stats as transceiver_stats
from BRIDGE.utils.meter import TimeMeter
seed_everything(42)

LATENCY_RESULTS_FILE = "experiments/Latency/latency_results.json"

time_meter = TimeMeter()


def get_prompt_dataset(total=2):
    with open("datasets/prompts/prompts.json", "r") as f:
        prompts = json.load(f)[: total]
    return prompts


class LatencyConfig(BRIDGEConfig):
    class evaluator:
        output_dir = F(str, default="outputs/", help="Output directory")
        max_new_tokens = F(int, default=100, help="Maximum number of tokens to generate")
        prompt_template = F(str, default="Context: {context}.\nInstruction: {query}\nAnswer: ", help="Prompt template")
        n_prompts = F(int, default=2, help="Number of prompts to evaluate")


class LatencyEvaluator(Evaluator):

    def __init__(self, config: LatencyConfig, baseline_name="DRAGON", stats_dict={}):
        super().__init__(config, name="Latency")
        self.name = "Latency"
        transceiver_stats.new_record()
        self.device = BRIDGE(config)
        self.prompt_template = config.evaluator.prompt_template
        self.max_new_tokens = config.evaluator.max_new_tokens
        self.dataset = get_prompt_dataset(config.evaluator.n_prompts)
        self.stats_dict = stats_dict
        self.baseline_name = baseline_name

    def set_stats_dict(self, latency: str, stats_file: str):
        if latency not in self.stats_dict:
            self.stats_dict[latency] = {}
        self.stats_dict[latency][self.baseline_name] = f"{stats_file}"
        with open(LATENCY_RESULTS_FILE, "w") as f:
            json.dump(self.stats_dict, f, indent=4)

    def _evaluate(self, latency: str):
        while not self.device.ready_for_generation:
            time.sleep(0.1)
        for item in tqdm(self.dataset):
            print(item['query'])
            with time_meter.timer("generation"):
                output_txt = self.device.query(item['query'], self.prompt_template, self.max_new_tokens)
            self.device.stats.update(time_meter.timer("generation"))
            print("-" * 50)
            print("Input:", item['query'], "\n\n")
            print("Output:", output_txt, "\n\n")
            print("-" * 50)
            transceiver_stats.new_record()

        run_output_dir = Path(self.config.output_dir, self.run_id)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        stats_file = run_output_dir / "stats.json"

        from BRIDGE.aggregator import stats as aggregator_stats
        (self.device.stats | aggregator_stats).dump(stats_file)
        transceiver_stats.dump(run_output_dir / "transceiver_stats.json")
        self.logger.debug(f"Stats saved to `{stats_file}`")
        self.set_stats_dict(latency, str(stats_file))
        # self.device.shutdown()

    def _clear_stats(self):
        from BRIDGE.aggregator import stats as aggregator_stats
        aggregator_stats.clear()
        transceiver_stats.clear()
        self.device.stats.clear()

    def evaluate(self):
        try:
            for latency in range(10, 311, 50):
                jitter = latency // 5
                transceiver.LATENCY = latency
                transceiver.JITTER = jitter
                print(f"Evaluating with latency: {latency} ms")
                print(f"Setting tc netem delay to {latency}ms ± {jitter}ms")
                self.run_id = self.name + "-" + datetime.now().strftime("%Y%m%d%H%M%S")
                self._evaluate(str(latency))
                time.sleep(5)
                self._clear_stats()
            self.device.shutdown()
        except Exception as e:
            self.logger.error(f"Evaluation failed with exception: {e}")
            self.device.shutdown()


if __name__ == "__main__":
    config = LatencyConfig()
    config.parse_sys_args()
    config.generator.s_sequence = 1024
    config.retriever.s_context = 256
    config.sampler.do_sample = False
    # Port for sending data
    config.trans.tx_port = 7010
    # Port for receiving data
    config.trans.rx_port = 7011

    if config.trans.rank == 0:
        # Cloud
        # Remote host ip address
        config.trans.tx_host = "127.0.0.1"
        config.trans.tx_port, config.trans.rx_port = config.trans.rx_port, config.trans.tx_port
        cloud = BRIDGE(config)
    else:
        # Device
        # Remote host ip address
        config.trans.tx_host = "127.0.0.1"
        with open(LATENCY_RESULTS_FILE, "r") as f:
            stats_dict = json.load(f)
        evaluator = LatencyEvaluator(config, baseline_name=input("Enter baseline name: "), stats_dict=stats_dict)
        evaluator.evaluate()
