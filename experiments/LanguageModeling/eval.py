
import sys
sys.path.append(".")

from BRIDGE.utils.stable import seed_everything
seed_everything(42)

import math
from tqdm import tqdm
from BRIDGE.config import BRIDGEConfig
from BRIDGE.utils.configure import Field as F
from BRIDGE.baselines.centralized_rag import DualModelRagForGeneration
from BRIDGE.utils.data_process.data_utils import ContextualLMSampleLoader
from experiments.evaluator import Evaluator
from experiments.metrics import CrossEntropy
from experiments.utils import load_dataset


class LanguageModelingConfig(BRIDGEConfig):
    class evaluator:
        dataset    = F(str,  required=True, help="Path to the dataset file")
        data_ratio = F(float ,default=1.0,   help="Ratio of the dataset to use")
        max_texts  = F(int,  default=None,  help="If set, only use the first N text records from the dataset")
        output_dir = F(str,  required=True, help="Path to save the evaluation output")
        s_block    = F(int,  default=10000, help="Number of documents to process in a block")
        s_prefix   = F(int,  default=128,   help="Size of the prefix in a rolling window of the test text")
        model1 = F(str, default="models/Qwen/Qwen2.5-1.5B", help="Baseline model 1 for comparison")
        model2 = F(str, default="models/Qwen/Qwen2.5-1.5B", help="Baseline model 2 for comparison")


class LanguageModelingEvaluator(Evaluator):

    def __init__(self, config: LanguageModelingConfig):
        super().__init__(config, name="LanguageModeling")
        self.rag = DualModelRagForGeneration(config)
        repo_id, dataset_id = config.evaluator.dataset.split(",")
        self.data = load_dataset(
            repo_id, dataset_id, cache_path=config.cache.directory,
            split=f"test[0%:{int(config.evaluator.data_ratio * 100)}%]"
        )["text"]
        if config.evaluator.max_texts is not None:
            self.data = self.data[: int(config.evaluator.max_texts)]
        self.tokenizer = self.rag.generator.tokenizer
        self.metric = CrossEntropy(device=config.device)
        self.template = "{context}{query}{input_text}"

    def data_loader(self):
        for i in tqdm(range(0, len(self.data), self.config.s_block)):
            texts = "\n\n".join(self.data[i: i + self.config.s_block])
            loader, total = ContextualLMSampleLoader(
                token_list=self.tokenizer.encode_plus(texts)["input_ids"],
                bos_token=self.rag.generator.context_switching_id,
                max_seq_len=self.rag.generator.max_seq_len,
                prefix_len=self.config.s_prefix)
            yield from tqdm(loader(), total=total)

    def evaluate(self):
        self.metric.reset()
        for query_ids, input_ids, label_ids in self.data_loader():
            if input_ids[0] is None or len(query_ids) > 256:
                continue
            logprobs = self.rag(query_ids, input_ids, template=self.template)
            self.metric.update(logprobs=logprobs, labels=label_ids)
        result = float(self.metric.compute())
        self.logger.info(f"{self.metric.__class__.__name__}: {result:.4f}")
        self.save_output({
            "cross_entropy": result,
            "perplexity": math.exp(result)
        })
        if retriever := self.rag.retriever:
            retriever.query2docs.flush()


if __name__ == "__main__":
    config = LanguageModelingConfig()
    config.parse_sys_args()
    evaluator = LanguageModelingEvaluator(config)
    evaluator.evaluate()
