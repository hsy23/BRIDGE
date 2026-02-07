from pathlib import Path
import sys
sys.path.append(".")

from tqdm import tqdm
from BRIDGE.config import BRIDGEConfig
from BRIDGE.utils.stable import seed_everything
from BRIDGE.utils.configure import Field as F
from experiments.evaluator import Evaluator
from BRIDGE.baselines.centralized_rag import RagTokenForGeneration
from BRIDGE.utils.meter import TimeMeter

time_meter = TimeMeter()


seed_everything(42)


def get_prompt_dataset(total=2):
    with open("datasets/prompts/prompts.json", "r") as f:
        prompts = f.readlines()[: total]
    return prompts


class LatencyConfig(BRIDGEConfig):
    class evaluator:
        output_dir = F(str, default="outputs/", help="Output directory")
        max_new_tokens = F(int, default=100, help="Maximum number of tokens to generate")
        prompt_template = F(str, default="Context: {context}.\nInstruction: {query}\nAnswer: ", help="Prompt template")
        n_prompts = F(int, default=2, help="Number of prompts to evaluate")


class LatencyEvaluator(Evaluator):

    def __init__(self, config: LatencyConfig):
        super().__init__(config, name="Latency")
        self.rag = RagTokenForGeneration(config)
        self.tokenizer = self.rag.generator.tokenizer
        self.max_new_tokens = config.evaluator.max_new_tokens
        self.prompt_template = config.evaluator.prompt_template
        self.dataset = get_prompt_dataset(config.evaluator.n_prompts)

    def evaluate(self):
        for query in tqdm(self.dataset):
            with time_meter.timer("generation"):
                query_ids = self.tokenizer.encode(query)
                output_ids, _, passage0 = self.rag.generate(
                    query_ids, max_new_tokens=self.max_new_tokens, template=self.prompt_template)
            self.rag.stats.update(time_meter.timer("generation"))

        run_output_dir = Path(self.config.output_dir, self.run_id)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        stats_file = run_output_dir / "stats.json"
        self.rag.stats.dump(stats_file)
        self.logger.info(f"Stats saved to `{stats_file}`")


if __name__ == "__main__":
    config = LatencyConfig()
    config.parse_sys_args()
    config.generator.s_sequence = 1024
    config.retriever.s_context = 256
    config.sampler.do_sample = False
    config.trans.rank = 1

    evaluator = LatencyEvaluator(config)
    evaluator.evaluate()
