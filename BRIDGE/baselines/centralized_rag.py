import torch
from typing import List, Tuple, Optional, Dict
from sentence_transformers import CrossEncoder
from transformers.modeling_outputs import CausalLMOutputWithPast
from tqdm import tqdm

from ..config import BRIDGEConfig
from ..generator import Generator
from ..utils.mlogging import Logger
from ..utils.meter import Statistics, TimeMeter


logger = Logger.build(__name__, level="INFO")
time_meter = TimeMeter()


def group_docs(docs: List[str], scores: List[float], s_aggregate: int) -> Tuple[List[str], List[float]]:
    """
    Group documents into s_aggregate number of documents.
    """
    chunk_size = len(docs) // s_aggregate
    if chunk_size == 0:
        logger.info(f"{len(scores)=} {s_aggregate=}")
    chunk_scores, chunk_docs = [], []
    for i in range(s_aggregate):
        beg, end = i * chunk_size, (i + 1) * chunk_size
        chunk_docs.append('\n'.join(docs[beg: end]))
        chunk_scores.append(sum(scores[beg: end]) / chunk_size)
    return chunk_docs, chunk_scores


class RagForGeneration:

    def __init__(self, config: BRIDGEConfig):
        self.config = config
        self.generator = Generator(config)
        self.retriever = None
        self.do_retrieval = config.retriever.n_docs > 0
        self.device = torch.device(config.device)
        self.context_size: int = config.retriever.s_context
        self.aggregate_size: int = config.retriever.s_aggregate

        if self.do_retrieval:
            if "wikitext" in config.retriever.passages:
                from ..retriever import CustomRetriever as Retriever
            elif config.retriever.passages == "wikipedia[local]":
                from ..retriever import DPRRetriever as Retriever
            elif config.retriever.passages == "wikipedia[remote]":
                from ..retriever import DPRRetrieverClient as Retriever
            self.retriever = Retriever(config)
            self.retriever.prepare_retrieval(config)

    def _prepare_inputs_for_generation(
        self, query_ids: List[int],
        input_ids: List[int],
        template: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.generator.tokenizer.decode(query_ids)
        input_text = self.generator.tokenizer.decode(input_ids)
        with time_meter.timer("Retrieval"):
            if self.config.retriever.downsample_type != 0:
                self.retriever.n_docs = self.config.retriever.n_docs * 2
                passages, scores = self.retriever.retrieve_passages([query])[0]
                if self.config.retriever.downsample_type == 1:
                    docs, scores = passages[: self.config.retriever.n_docs], scores[: self.config.retriever.n_docs]
                else:
                    docs, scores = passages[-self.config.retriever.n_docs:], scores[-self.config.retriever.n_docs:]
            else:
                docs, scores = self.retriever.retrieve_passages([query])[0]

        logger.debug(f"Retrieval complete in {time_meter.timer('Retrieval').duration:.4f} seconds.")

        with time_meter.timer("Tokenization"):
            doc_texts = [doc["text"] for doc in docs]
            doc_texts, scores = group_docs(doc_texts, scores, self.aggregate_size)

            doc_texts = [
                self.generator.tokenizer.decode(self.generator.tokenizer.encode(doc_text)[: self.context_size])
                for doc_text in doc_texts
            ]
            input_text_list = [
                template.format(context=doc_text, query=query, input_text=input_text)
                for doc_text in doc_texts
            ]

            inputs_encoding = self.generator.tokenizer.batch_encode_plus(
                input_text_list, padding='longest', return_tensors='pt')
            context_input_ids = inputs_encoding['input_ids']
            context_input_ids = context_input_ids.to(self.device)
            attention_mask = inputs_encoding['attention_mask'].to(self.device)

            scores = scores[: self.aggregate_size]
            scores = torch.as_tensor(scores, dtype=torch.float32)
            scores = torch.nn.functional.log_softmax(scores, dim=-1)
            scores = scores.to(self.device)
        logger.debug(f"Tokenization complete in {time_meter.timer('Tokenization').duration:.4f} seconds.")
        return context_input_ids, attention_mask, scores, doc_texts

    def __call__(self, query_ids: List[int], input_ids: List[int], template: str) -> torch.Tensor:
        """
        Given query_ids and input_ids, return the next token logprobs of each input_id.
        """
        if not self.do_retrieval or len(query_ids) == 0:
            input_ids = torch.as_tensor(query_ids + input_ids, dtype=torch.long).to(self.device).unsqueeze(0)
            output = self.generator(input_ids=input_ids, attention_mask=torch.ones_like(input_ids).to(self.device))
            logprobs = torch.log_softmax(output.logits[0, len(query_ids):], dim=-1)
            return logprobs
        context_input_ids, attention_mask, scores, doc_texts = self._prepare_inputs_for_generation(query_ids, input_ids, template)
        _, logprobs, _ = self._generate(context_input_ids, attention_mask, scores, n_logits=len(input_ids))
        return logprobs

    def _aggregate_outputs(self, output_1, output_2, n_logits, scores):
        logits = torch.vstack([output_1.logits[:, -n_logits:], output_2.logits[:, -n_logits:]])
        del output_1.logits, output_2.logits
        logprobs = torch.nn.functional.log_softmax(     # (s_aggregate, s_sequence, s_vocab)
            logits / self.generator.sampler.temperature, dim=-1)
        del logits
        logprobs = logprobs.permute(1, 0, 2)            # (s_sequence, s_aggregate, s_vocab)
        logprobs.add_(scores.unsqueeze(dim=-1))  # (s_sequence, s_aggregate, s_vocab) + (s_aggregate, 1)
        logprobs = torch.logsumexp(logprobs, dim=1)
        next_token = self.generator.sampler(torch.exp(logprobs[-1]))
        return next_token, logprobs

    def _generate(
            self, input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            scores: torch.Tensor, n_logits: int,
            past_key_values=None) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Prefill concatenations of context_ids and input_ids, generate the next token, and return the logprobs
        during the prefilling process.

        Output Aggregation:
        probs = softmax(w)^T softmax(z) -> log(probs) = logsumexp(logsoftmax(w)+logsoftmax(z))
        """
        with time_meter.timer("Decoding"):
            if input_ids.shape[0] == 1:  # context aggregation
                output = self.generator(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values
                )
                logprobs = torch.nn.functional.log_softmax(output.logits[:, -n_logits:], dim=-1).squeeze(dim=0)
                next_token = self.generator.sampler(torch.exp(logprobs[-1]))
                return next_token, logprobs, output.past_key_values
            half_size = input_ids.shape[0] // 2
            # past_key_values_1 = [[], []] if past_key_values is not None else None
            # past_key_values_2 = [[], []] if past_key_values is not None else None
            # if past_key_values is not None:
            #     for layer_idx in range(len(past_key_values)):
            #         past_key_values_1[0].append(past_key_values[layer_idx][0][: half_size])
            #         past_key_values_1[1].append(past_key_values[layer_idx][1][: half_size])
            #         past_key_values_2[0].append(past_key_values[layer_idx][0][half_size: ])
            #         past_key_values_2[1].append(past_key_values[layer_idx][1][half_size: ])
            past_key_values_1 = past_key_values_2 = None
            if past_key_values is not None:
                past_key_values_1, past_key_values_2 = past_key_values

            output_1 = self.generator(
                input_ids=input_ids[: half_size],
                attention_mask=attention_mask[: half_size],
                past_key_values=past_key_values_1
            )
            output_2 = self.generator(
                input_ids=input_ids[half_size:],
                attention_mask=attention_mask[half_size:],
                past_key_values=past_key_values_2
            )
            # past_key_values = [[], []]
            # for layer_idx in range(len(output_1.past_key_values)):
            #     past_key_values[0].append(torch.vstack([output_1.past_key_values[layer_idx][0], output_2.past_key_values[layer_idx][0]]))
            #     past_key_values[1].append(torch.vstack([output_1.past_key_values[layer_idx][1], output_2.past_key_values[layer_idx][1]]))
            past_key_values = (output_1.past_key_values, output_2.past_key_values)
            next_token, logprobs = self._aggregate_outputs(output_1, output_2, n_logits, scores)
        logger.debug(f"Decoding complete in {time_meter.timer('Decoding').duration:.4f} seconds.")
        return next_token, logprobs, past_key_values


class DualModelRagForGeneration(RagForGeneration):

    def __init__(self, config):
        super().__init__(config)
        config.generator.model = config.evaluator.model1
        self.generator1 = Generator(config)
        config.generator.model = config.evaluator.model2
        self.generator2 = Generator(config)

        self._base_vocab_size = int(len(self.generator.tokenizer))
        self._vocab_alignment: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
        self._vocab_alignment["gen1"] = self._build_vocab_alignment(self.generator, self.generator1)
        self._vocab_alignment["gen2"] = self._build_vocab_alignment(self.generator, self.generator2)

    def _build_vocab_alignment(self, base_generator: Generator, other_generator: Generator) -> Dict[str, Optional[torch.Tensor]]:
        base_vocab_size = int(len(base_generator.tokenizer))
        other_vocab_size = int(len(other_generator.tokenizer))

        if other_vocab_size >= base_vocab_size and self._is_prefix_identity(base_generator, other_generator, base_vocab_size):
            return {"mode": "identity", "base_to_other": None}

        base_to_other = torch.full((base_vocab_size,), -1, dtype=torch.long)
        base_tok = base_generator.tokenizer
        other_tok = other_generator.tokenizer
        for base_id in range(base_vocab_size):
            token = base_tok.convert_ids_to_tokens(base_id)
            if token is None:
                continue
            other_id = other_tok.convert_tokens_to_ids(token)
            if isinstance(other_id, list):
                other_id = other_id[0] if other_id else -1
            if isinstance(other_id, int) and 0 <= other_id:
                base_to_other[base_id] = other_id
        return {"mode": "map", "base_to_other": base_to_other}

    def _is_prefix_identity(self, base_generator: Generator, other_generator: Generator, base_vocab_size: int) -> bool:
        base_tok = base_generator.tokenizer
        other_tok = other_generator.tokenizer
        other_vocab_size = int(len(other_tok))
        if other_vocab_size < base_vocab_size:
            return False
        sample_ids = [0, 1, 2, 3, 4, 5, 10, 50, 100, 500, 1000]
        sample_ids += [max(0, base_vocab_size // 4), max(0, base_vocab_size // 2), max(0, (3 * base_vocab_size) // 4)]
        sample_ids += [max(0, base_vocab_size - 1), max(0, base_vocab_size - 2), max(0, base_vocab_size - 3)]
        sample_ids = sorted(set(i for i in sample_ids if 0 <= i < base_vocab_size))
        for idx in sample_ids:
            tok = base_tok.convert_ids_to_tokens(idx)
            if tok is None:
                return False
            if other_tok.convert_ids_to_tokens(idx) != tok:
                return False
            other_id = other_tok.convert_tokens_to_ids(tok)
            if isinstance(other_id, list):
                other_id = other_id[0] if other_id else -1
            if other_id != idx:
                return False
        return True

    def _convert_input_ids(self, input_ids: torch.LongTensor, alignment: Dict[str, Optional[torch.Tensor]], other_generator: Generator) -> torch.LongTensor:
        if alignment["mode"] == "identity":
            return input_ids

        base_to_other = alignment["base_to_other"]
        assert base_to_other is not None
        base_to_other = base_to_other.to(device=input_ids.device)

        mapped = base_to_other[input_ids]
        fallback_id = getattr(other_generator.model.config, "eos_token_id", None)
        if fallback_id is None:
            fallback_id = 0
        mapped = torch.where(mapped >= 0, mapped, torch.full_like(mapped, int(fallback_id)))
        return mapped

    def _align_logits_to_base(self, logits: torch.Tensor, alignment: Dict[str, Optional[torch.Tensor]], other_generator: Generator) -> torch.Tensor:
        base_vocab_size = self._base_vocab_size
        other_vocab_size = int(logits.size(-1))

        if alignment["mode"] == "identity":
            if other_vocab_size >= base_vocab_size:
                return logits[..., :base_vocab_size]
            padded = logits.new_full(logits.shape[:-1] + (base_vocab_size,), float("-inf"))
            padded[..., :other_vocab_size] = logits
            return padded

        base_to_other = alignment["base_to_other"]
        assert base_to_other is not None
        base_to_other = base_to_other.to(device=logits.device)
        index = base_to_other.clone()
        invalid = index < 0
        index[invalid] = 0
        aligned = logits.index_select(dim=-1, index=index)
        if invalid.any():
            aligned[..., invalid] = float("-inf")
        return aligned

    def _aggregate_outputs(self, output_1: CausalLMOutputWithPast, output_2: CausalLMOutputWithPast, n_logits: int, scores: torch.Tensor):
        logits_1 = output_1.logits[:, -n_logits:]
        logits_2 = output_2.logits[:, -n_logits:]
        logits_1 = self._align_logits_to_base(logits_1, self._vocab_alignment["gen1"], self.generator1)
        logits_2 = self._align_logits_to_base(logits_2, self._vocab_alignment["gen2"], self.generator2)

        logits = torch.vstack([logits_1, logits_2])
        del output_1.logits, output_2.logits
        logprobs = torch.nn.functional.log_softmax(     # (s_aggregate, s_sequence, s_vocab)
            logits / self.generator.sampler.temperature, dim=-1)
        del logits
        logprobs = logprobs.permute(1, 0, 2)            # (s_sequence, s_aggregate, s_vocab)
        logprobs.add_(scores.unsqueeze(dim=-1))  # (s_sequence, s_aggregate, s_vocab) + (s_aggregate, 1)
        logprobs = torch.logsumexp(logprobs, dim=1)
        next_token = self.generator.sampler(torch.exp(logprobs[-1]))
        return next_token, logprobs

    def _generate(
            self, input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            scores: torch.Tensor, n_logits: int,
            past_key_values=None) -> Tuple[int, torch.Tensor, torch.Tensor]:
        with time_meter.timer("Decoding"):
            if input_ids.shape[0] == 1:  # context aggregation
                output = self.generator(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values
                )
                logprobs = torch.nn.functional.log_softmax(output.logits[:, -n_logits:], dim=-1).squeeze(dim=0)
                next_token = self.generator.sampler(torch.exp(logprobs[-1]))
                return next_token, logprobs, output.past_key_values
            half_size = input_ids.shape[0] // 2

            past_key_values_1 = past_key_values_2 = None
            if past_key_values is not None:
                past_key_values_1, past_key_values_2 = past_key_values

            input_ids_1 = self._convert_input_ids(input_ids[: half_size], self._vocab_alignment["gen1"], self.generator1)
            input_ids_2 = self._convert_input_ids(input_ids[half_size:], self._vocab_alignment["gen2"], self.generator2)

            output_1 = self.generator1(
                input_ids=input_ids_1,
                attention_mask=attention_mask[: half_size],
                past_key_values=past_key_values_1
            )
            output_2 = self.generator2(
                input_ids=input_ids_2,
                attention_mask=attention_mask[half_size:],
                past_key_values=past_key_values_2
            )
            past_key_values = (output_1.past_key_values, output_2.past_key_values)
            next_token, logprobs = self._aggregate_outputs(output_1, output_2, n_logits, scores)
        logger.debug(f"Decoding complete in {time_meter.timer('Decoding').duration:.4f} seconds.")
        return next_token, logprobs, past_key_values


class RagTokenForGeneration(RagForGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.do_rerank = config.reranker.do_rerank
        if self.do_rerank:
            self.rerank_model = CrossEncoder(
                model_name=config.reranker.model,
                max_length=512, device=config.device)
            self.rerank_momentum = config.reranker.momentum
        self.stats = Statistics()

    def _rerank_documents(
        self, query_ids: List[int],
        past_generated: List[int],
        doc_texts: List[str],
        scores: torch.Tensor
    ) -> torch.Tensor:
        new_query_ids = query_ids + past_generated
        new_query = self.generator.tokenizer.decode(new_query_ids)

        with time_meter.timer("Re-ranking"):
            pairs = [[new_query, doc] for doc in doc_texts]
            new_scores = self.rerank_model.predict(
                pairs, activation_fct=torch.sigmoid,
                batch_size=len(pairs), convert_to_numpy=False
            )
        logger.debug(f"Re-ranking complete in {time_meter.timer('Re-ranking').duration:.4f} seconds.")

        new_scores = torch.nn.functional.log_softmax(torch.as_tensor(new_scores), dim=-1).to(self.device)
        new_scores = scores * self.rerank_momentum + new_scores * (1 - self.rerank_momentum)
        return new_scores

    def generate(
        self,
        query_ids: List[int],
        max_new_tokens: int,
        template: str
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Prefill the context that consists of retrieved passages and the query_ids, and generate max_new_tokens
        tokens autoregressively. Return the newly generated tokens and their corresponding logprobs.

        Aggregate tokens from multiple generation processes at each step to obtain the next token. Repeat this
        autoregressive generation process for max_new_tokens times.
        """
        self.stats.new_record()
        scores_list = []  # for debugging
        pbar = tqdm(total=max_new_tokens, desc="Generating", leave=False, initial=0)
        context_input_ids, attention_mask, scores, doc_texts = self._prepare_inputs_for_generation(query_ids, [], template)
        # with open("logs/docs.pkl", "wb") as f:
        #     pickle.dump(doc_texts, f)
        scores_list.append(scores)

        # Pre-fill the context
        next_token, logprobs, past_key_values = self._generate(context_input_ids, attention_mask, scores, n_logits=1)
        logprobs, output_ids = [logprobs[0]], [next_token]
        pbar.update(1)

        for t in range(max_new_tokens - 1):  # Auto-regressive generation
            # Dynamic re-ranking every `self.config.reranker.period` steps
            with time_meter.timer("LatencyPerToken"):
                if self.do_rerank and (self.config.reranker.period == 0 or (t + 1) % self.config.reranker.period == 0):
                    scores = self._rerank_documents(query_ids, output_ids, doc_texts, scores)
                scores_list.append(scores)
                input_ids = torch.as_tensor([next_token], dtype=torch.long).to(self.device)
                input_ids = input_ids.repeat(self.aggregate_size, 1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=1)
                next_token, logprobs_token_wise, past_key_values = self._generate(
                    input_ids, attention_mask, scores, n_logits=1, past_key_values=past_key_values)
                logprobs.append(logprobs_token_wise)
                output_ids.append(next_token)
            self.stats.update(time_meter.timer('LatencyPerToken'))
            pbar.update(1)
        pbar.close()
        logprobs = torch.vstack(logprobs)    # (max_new_tokens, s_vocab)
        scores_list = torch.vstack(scores_list).exp()  # (max_new_tokens, s_aggregate)
        # with open("logs/scores.pkl", "wb") as f:
        #     pickle.dump(scores_list.cpu().tolist(), f)
        return output_ids, logprobs, doc_texts


class CoGenRagTokenForGeneration(RagForGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.do_rerank = config.reranker.do_rerank
        if self.do_rerank:
            self.rerank_model = CrossEncoder(
                model_name=config.reranker.model,
                max_length=512, device=config.device)
            self.rerank_momentum = config.reranker.momentum
        self.stats = Statistics()

    def _rerank_documents(
            self, query_ids: List[int],
            past_generated: List[int],
            doc_texts: List[str],
            scores: torch.Tensor
    ) -> torch.Tensor:
        new_query_ids = query_ids + past_generated
        new_query = self.generator.tokenizer.decode(new_query_ids)

        with time_meter.timer("Re-ranking"):
            pairs = [[new_query, doc] for doc in doc_texts]
            new_scores = self.rerank_model.predict(
                pairs, activation_fct=torch.sigmoid,
                batch_size=len(pairs), convert_to_numpy=False
            )
        logger.debug(f"Re-ranking complete in {time_meter.timer('Re-ranking').duration:.4f} seconds.")

        new_scores = torch.nn.functional.log_softmax(torch.as_tensor(new_scores), dim=-1).to(self.device)
        new_scores = scores * self.rerank_momentum + new_scores * (1 - self.rerank_momentum)
        return new_scores

    def query(
            self,
            query: str,
            max_new_tokens: int
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Prefill the context that consists of retrieved passages and the query_ids, and generate max_new_tokens
        tokens autoregressively. Return the newly generated tokens and their corresponding logprobs.

        Aggregate tokens from multiple generation processes at each step to obtain the next token. Repeat this
        autoregressive generation process for max_new_tokens times.
        """
        output_ids, logprobs, _ = self._generate(query, max_new_tokens=max_new_tokens)
        return self.generator.tokenizer.decode(output_ids, skip_special_tokens=True)

    def _generate(
            self,
            prompt: str,
            max_new_tokens: int
    ) -> Tuple[List[int], torch.Tensor, List[str]]:
        """Generate tokens from a fully-formed prompt (no retrieval step).

        This baseline is used by experiments/CoGen where the prompt already contains
        all context (profile/history/public context). Returns only newly generated
        token ids (excluding the prompt tokens).
        """
        self.stats.new_record()

        pbar = tqdm(total=max_new_tokens, desc="Generating", leave=False, initial=0)
        inputs_encoding = self.generator.tokenizer.batch_encode_plus(
            [prompt], padding='longest', return_tensors='pt'
        )
        input_ids = inputs_encoding['input_ids'].to(self.device)
        attention_mask = inputs_encoding['attention_mask'].to(self.device)

        output_ids: List[int] = []
        logprobs_list: List[torch.Tensor] = []

        past_key_values = None
        next_token = None

        # Prefill
        with time_meter.timer("Decoding"):
            output = self.generator(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            past_key_values = output.past_key_values
            step_logprobs = torch.nn.functional.log_softmax(output.logits[0, -1], dim=-1)
            next_token = self.generator.sampler(torch.exp(step_logprobs))
        logprobs_list.append(step_logprobs)
        output_ids.append(next_token)
        pbar.update(1)

        for _ in range(max_new_tokens - 1):
            with time_meter.timer("LatencyPerToken"):
                step_input_ids = torch.as_tensor([[next_token]], dtype=torch.long).to(self.device)
                attention_mask = torch.cat([attention_mask, torch.ones_like(step_input_ids)], dim=1)
                output = self.generator(
                    input_ids=step_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
                past_key_values = output.past_key_values
                step_logprobs = torch.nn.functional.log_softmax(output.logits[0, -1], dim=-1)
                next_token = self.generator.sampler(torch.exp(step_logprobs))

            self.stats.update(time_meter.timer('LatencyPerToken'))
            logprobs_list.append(step_logprobs)
            output_ids.append(next_token)
            pbar.update(1)

        pbar.close()
        logprobs = torch.stack(logprobs_list, dim=0)  # (max_new_tokens, vocab)
        return output_ids, logprobs, []
