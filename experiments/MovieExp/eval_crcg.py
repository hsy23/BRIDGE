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
from BRIDGE.baselines.centralized_rag import CoGenRagTokenForGeneration

sys.path.append(".")


generator.logging_level = "DEBUG"
transceiver.logging_level = "DEBUG"
decoder.logging_level = "DEBUG"
bridge.logging_level = "DEBUG"
aggregator.logging_level = "DEBUG"
time_meter = TimeMeter()

LOGGER = Logger.build(__name__, level="INFO")

SYSTEM_PROMPT = (
  "You are a highly knowledgeable manager responsible for guiding a subordinate "
  "who has access to user data you cannot view. Your task is to help them craft a "
  "brief, compelling explanation for why a specific movie is recommended to a user. "
  "Provide concise, actionable guidelines with several clear points. "
  "Instruct the subordinate to thoughtfully leverage available user profile details "
  "(age, sex, occupation, current city, birth city, education, income, relationship status) "
  "and the user’s preferred style and tone. "
  "Emphasize personalization: align the recommendation with the user’s unique interests, "
  "lifestyle, cultural context, and life stage. "
  "Keep the final explanation short, engaging, and benefit-focused, avoiding spoilers "
  "and unnecessary plot summary. "
  "Aim for insightful, high-quality guidance that results in a recommendation the user "
  "finds relevant, appealing, and satisfying."
  "Don't repeat the Public Context(Additional Information) in your response. "
)

MovieExp_CONTEXT_TEMPLATE = """## User Profile(Additional Information)
{profile}
"""

PUBLIC_CONTEXT_TEMPLATE = """## Public Context(Additional Information)
{public_context}
"""

DEVICE_USER_TEMPLATE = """## Task
{task}

{context}

## Instruction
{extra} The output should be natural, coherent, and context-aware. Do not include any self-introduction, persona framing, or first-person language. Do not restate or explain the task, and do not mention the template, instructions, or reasoning process. Keep the response concise yet sufficiently informative. Avoid numbered lists, bullet points, citations, or any meta-commentary.

## Response
"""


class MovieExpConfig(BRIDGEConfig):
    class evaluator:
        output_dir = F(str, default="outputs/MovieExp_", help="Output directory")
        data_path = F(str, default="/workspace/LSRP/data/MovieExp/test.json", help="MovieExp test.json path")
        n_prompts = F(int, default=10, help="Number of prompts per setting")
        max_new_tokens = F(int, default=256, help="Tokens per response")
        seed = F(int, default=46, help="Random seed")
        wiki = F(bool, default=False, help="Use Wikipedia passages")
        movieexp = F(bool, default=False, help="Use MovieExp profile passages")


def load_MovieExp(path: str, limit: int):
    with open(path, "r") as f:
        data = json.load(f)
    return random.sample(data, limit) if limit > 0 else data


def build_profile_passages(sample, max_entries=4):
    passages = []
    for entry in sample.get("profile", [])[:max_entries]:
        title = (entry.get("title") or "").strip()
        body = (entry.get("text") or "").strip()
        if not body and not title:
            continue
        text = f"{title}\n{body}" if title else body
        passages.append(text)
    return passages


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


class MovieExpEvaluator(Evaluator):
    def __init__(self, config: MovieExpConfig):
        super().__init__(config, name="MovieExp")
        self.full_config = config
        self.dataset = load_MovieExp(config.evaluator.data_path, config.evaluator.n_prompts)
        self.rag = CoGenRagTokenForGeneration(config)
        self.retriever = None
        self.wiki_enable = self.full_config.evaluator.wiki
        self.MovieExp_enable = self.full_config.evaluator.movieexp
        self.stats = Statistics()

        if self.wiki_enable:
            config.retriever.n_docs = 2
            if "wikitext" in config.retriever.passages:
                from BRIDGE.retriever import CustomRetriever as Retriever
            elif config.retriever.passages == "wikipedia[local]":
                from BRIDGE.retriever import DPRRetriever as Retriever
            elif config.retriever.passages == "wikipedia[remote]":
                from BRIDGE.retriever import DPRRetrieverClient as Retriever
            self.retriever = Retriever(config)
            self.retriever.prepare_retrieval(config)

    def build_device_prompt(self, sample):
        context = ""
        if self.wiki_enable:
            docs, scores = self.retriever.retrieve_passages([sample["conversations"][0]["value"]])[0]
            doc_texts = [doc["text"] for doc in docs]
            doc_texts = [
                self.rag.generator.tokenizer.decode(self.rag.generator.tokenizer.encode(doc_text)[: self.rag.context_size])
                for doc_text in doc_texts
            ]
            context += PUBLIC_CONTEXT_TEMPLATE.format(public_context="\n\n".join(doc_texts))
        if self.MovieExp_enable:
            print("Building MovieExp context passages...")
            if context:
                context += "\n\n"
            context += MovieExp_CONTEXT_TEMPLATE.format(
                profile=sample.get("additional_profile", ""),
                task=sample["conversations"][0]["value"],
            )
        return (SYSTEM_PROMPT if self.MovieExp_enable else "") + "\n\n" + DEVICE_USER_TEMPLATE.format(
            context=context,
            task=sample["conversations"][0]["value"],
            extra="Use the User Profile and Chat History to infer the user’s preferences, style, and tone about the movie, and complete the Task accordingly. " if self.MovieExp_enable else "Use the Public Context to learn about the movie and complete the Task accordingly. ",
        )

    def evaluate(self):
        output_dir = Path(self.full_config.evaluator.output_dir, f"{self.run_id}-{self.full_config.generator.model.split('/')[-1]}-{'wiki' if self.wiki_enable else ''}{'MovieExp' if self.MovieExp_enable else ''}")
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        outputs = []

        for sample in tqdm(self.dataset, desc="Processing prompts"):
            task = sample["conversations"][0]["value"]
            # print(sample)
            prompt = self.build_device_prompt(sample)
            print(f"Prompt:\n{prompt}\n{'-'*80}\n")
            with time_meter.timer("TotalGenerationTime") as t_timer:
                output_text = self.rag.query(
                    prompt,
                    max_new_tokens=self.full_config.evaluator.max_new_tokens
                )
            run_time = t_timer.duration
            outputs.append(
                {
                    "tid": sample.get("tid"),
                    "task": task,
                    "profile": sample.get("additional_profile", ""),
                    "output": output_text,
                }
            )

        result = {
            "n_docs": self.full_config.retriever.n_docs,
            "generate_time": run_time,
            "outputs": outputs,
        }
        results.append(result)
        self.rag.stats.dump(output_dir / "stats.json")
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to `{output_dir / 'results.json'}`")


if __name__ == "__main__":
    config = MovieExpConfig()
    config.parse_sys_args()
    seed_everything(config.evaluator.seed)
    config.generator.s_sequence = 1024
    config.retriever.s_context = 256

    evaluator = MovieExpEvaluator(config)
    evaluator.evaluate()
