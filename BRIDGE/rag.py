from queue import Queue
from typing import List, Tuple, Union
import torch
from sentence_transformers import CrossEncoder
from transformers.cache_utils import DynamicCache
from .config import BRIDGEConfig
from .generator import CausalOutput, PreemptableGenerator
from .utils.meter import TimeMeter
from .utils.mlogging import Logger


logging_level = "INFO"
time_meter = TimeMeter()


class Rag:

    def __init__(self, config: BRIDGEConfig):
        self.rank_idx = config.trans.rank
        self.logger = Logger.build(__class__.__name__, logging_level)
        self.config = config
        self.device = torch.device(config.device)
        self.context_size: int = config.retriever.s_context  # need to be same across device and cloud
        self.aggregate_size: int = max(1, config.retriever.s_aggregate)
        self.do_retrieve = config.retriever.n_docs > 0
        self.do_rerank = config.reranker.do_rerank
        self.importance_sampling = (
            config.aggregator.mode == "speculative" and self.rank_idx == 1
        )
        self.input_queue = Queue(0)
        self.output_queue = Queue(0)
        self.local_context = None
        self.generator = PreemptableGenerator(
            config, self.input_queue, self.output_queue)
        self.generator.start()

        self.retriever = None
        if self.do_retrieve:
            passages = config.retriever.passages
            if "wikitext" in passages:
                from .retriever import CustomRetriever as Retriever
            elif passages == "wikipedia[local]":
                from .retriever import DPRRetriever as Retriever
            elif passages == "wikipedia[remote]":
                from .retriever import DPRRetrieverClient as Retriever

            self.retriever = Retriever(config)
            self.retriever.prepare_retrieval(config)

        self.reranker = None
        if self.do_rerank:
            self.rerank_model = CrossEncoder(
                model_name=config.reranker.model,
                max_length=512).to(config.device)
            self.rerank_momentum = config.reranker.momentum

    def _crop_context(self, context: str) -> str:
        return self.generator.tokenizer.decode(
            self.generator.tokenizer.encode(context)[: self.context_size]
        )

    def _is_chat_formatted(self, text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        if "[inst]" in lowered or "<|start_header_id|>" in lowered:
            return True
        return False

    def _wrap_chat_prompt(self, user_content: str) -> str:
        if self._is_chat_formatted(user_content):
            return user_content
        tokenizer = self.generator.tokenizer
        if not hasattr(tokenizer, "apply_chat_template"):
            return user_content
        # Some tokenizers expose apply_chat_template but have no chat_template set.
        # In that case, fall back to raw text to avoid ValueError.
        if not getattr(tokenizer, "chat_template", None):
            return user_content
        OUTPUT_GUARDRAIL = (
            "IMPORTANT: Output ONLY the final answer text. "
            "Do NOT include any preface, role/identity statements, meta commentary, analysis, "
            "headers, numbered lists, bullet points, or section labels. "
            "Do NOT mention that you are an AI, assistant, or copywriter."
        )
        system_content = (
            "You are now a helpful personal AI assistant. You should emulate the author's style "
            "and tone based on provided history content. Your responses should be detailed and "
            "informative, using the personal information reasonably in the user's profile. "
            "Aim for insightful and high-quality solutions that make users satisfied."
        )
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": OUTPUT_GUARDRAIL + '\n' + system_content},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _group_passages(self, passages: List[str], scores: List[float]) -> Tuple[List[str], List[float]]:
        """
        Partition the documents into groups of size self.aggregate_size,
        and calculate the average score for each group.
        """
        group_size = len(passages) // self.aggregate_size
        group_scores, group_passages = [], []
        for i in range(self.aggregate_size):
            beg, end = i * group_size, (i + 1) * group_size
            group_passages.append('\n'.join(passages[beg: end]))
            group_scores.append(sum(scores[beg: end]) / group_size)
        return group_passages, group_scores

    def _prepare_inputs_for_generation(
        self, query: str, prompt_template: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        if not self.do_retrieve:
            with time_meter.timer("Tokenization"):
                prompt = self.local_context or ""
                if "llama" in (self.config.generator.model or "").lower():
                    prompt = self._wrap_chat_prompt(prompt)
                scores = [100.0] * max(self.aggregate_size, 1)
                inputs_encoding = self.generator.tokenizer.batch_encode_plus(
                    [prompt], padding='longest', return_tensors='pt')
                input_ids = inputs_encoding['input_ids'].to(self.device)
                attention_mask = inputs_encoding['attention_mask'].to(self.device)
            scores = torch.as_tensor(scores, dtype=torch.float32, device=self.device)
            return input_ids, attention_mask, scores, [prompt]

        with time_meter.timer("Retrieval"):
            self.retriever.n_docs = self.config.retriever.n_docs * 2
            passages, scores = self.retriever.retrieve_passages([query])[0]
            if self.rank_idx == 0:
                passages, scores = passages[: self.config.retriever.n_docs], scores[: self.config.retriever.n_docs]
            else:
                passages, scores = passages[self.config.retriever.n_docs:], scores[self.config.retriever.n_docs:]
        self.logger.debug(f"Retrieval complete in {time_meter.timer('Retrieval').duration:.4f} seconds.")
        with time_meter.timer("Tokenization"):
            # assemble input from passages, query and prompt_template
            passages = [p["text"] for p in passages]
            passages, scores = self._group_passages(passages, scores)
            passages = [self._crop_context(passage) for passage in passages]
            input_text_list = [
                prompt_template.format(
                    context=passage,
                    query=query
                ) for passage in passages
            ]
            if "llama" in (self.config.generator.model or "").lower():
                input_text_list = [self._wrap_chat_prompt(text) for text in input_text_list]

            # encode input into input_ids and attention_mask
            inputs_encoding = self.generator.tokenizer.batch_encode_plus(
                input_text_list, padding='longest', return_tensors='pt')
            input_ids = inputs_encoding['input_ids'].to(self.device)
            attention_mask = inputs_encoding['attention_mask'].to(self.device)

        self.logger.debug(f"Tokenization complete in {time_meter.timer('Tokenization').duration:.4f} seconds.")
        scores = torch.as_tensor(scores, dtype=torch.float32, device=self.device)

        return input_ids, attention_mask, scores, passages

    def _rerank_passages(
        self, query: str, generated: str, passages: List[str], scores: torch.Tensor
    ) -> torch.Tensor:
        new_query = f"{query} {generated}"
        with time_meter.timer("Re-ranking"):
            pairs = [[new_query, doc] for doc in passages]
            new_scores = self.rerank_model.predict(
                pairs, activation_fct=torch.sigmoid,
                batch_size=len(pairs), convert_to_numpy=False
            )
        self.logger.debug(f"Re-ranking complete in {time_meter.timer('Re-ranking').duration:.4f} seconds.")

        new_scores = scores * self.rerank_momentum + new_scores * (1 - self.rerank_momentum)
        new_scores = torch.nn.functional.log_softmax(new_scores, dim=-1)
        return new_scores

    def rerank_passages(self, query: str, generated: str, passages: List[str], scores: torch.Tensor, step: int) -> torch.Tensor:
        # Dynamic re-ranking every `self.config.reranker.period` steps
        if self.do_rerank and (self.config.reranker.period == 0 or (step + 1) % self.config.reranker.period == 0):
            scores = self._rerank_passages(query, generated, passages, scores)
        return scores

    def _generate(
            self, input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            scores: torch.Tensor,
            past_key_values=None, preemptable=False) -> CausalOutput:
        """
        Prefill concatenations of context_ids and input_ids, generate the next token, and return the logprobs
        during the prefilling process.
        Output Aggregation:
        probs = softmax(w)^T softmax(z) -> log(probs) = logsumexp(logsoftmax(w)+logsoftmax(z))
        """
        # ipdb.set_trace()
        with time_meter.timer("Decoding"):
            if preemptable:
                self.input_queue.put((input_ids, attention_mask, {"past_key_values": past_key_values}))
                output = self.output_queue.get()

                if output is None:
                    return None
            else:
                output = self.generator(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values
                )

            past_key_values_out = output.past_key_values
            # Some models (e.g. Qwen) return a `DynamicCache`, but our scroll-back
            # logic expects a legacy tuple-of-tuples cache that can be sliced.
            if isinstance(past_key_values_out, DynamicCache):
                past_key_values_out = past_key_values_out.to_legacy_cache()
            logscores = torch.nn.functional.log_softmax(scores, dim=0)
            logprobs = torch.nn.functional.log_softmax(    # (s_aggregate, s_vocab)
                output.logits[:, -1] / self.generator.sampler.temperature, dim=-1)
            logprobs = logprobs.permute(1, 0)              # (s_vocab, s_aggregate)
            logprobs = logprobs + logscores                # (s_vocab, s_aggregate) + (s_aggregate,)
            logprobs = torch.logsumexp(logprobs, dim=1)    # (s_vocab,)
            if self.importance_sampling:
                next_token = torch.multinomial(torch.exp(logprobs), num_samples=1).squeeze().cpu().item()
            else:
                next_token = self.generator.sampler(torch.exp(logprobs))

        self.logger.debug(f"Decoding complete in {time_meter.timer('Decoding').duration:.4f} seconds.")

        weight = torch.logsumexp(scores, dim=0).item()
        return CausalOutput(
            next_token=next_token, logprobs=logprobs, weight=weight,
            past_key_values=past_key_values_out
        )

    def generate(self, last_token: Union[List, int], scores: torch.Tensor, attention_mask: torch.Tensor, past_key_values: List[torch.Tensor], preemptable=True) -> Tuple[CausalOutput, torch.Tensor]:
        if isinstance(last_token, int):
            last_token = [last_token]
        input_ids = torch.as_tensor(last_token, dtype=torch.long, device=self.device)
        input_ids = input_ids.repeat(self.aggregate_size, 1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=1)
        return self._generate(input_ids, attention_mask, scores, past_key_values=past_key_values, preemptable=preemptable), attention_mask
