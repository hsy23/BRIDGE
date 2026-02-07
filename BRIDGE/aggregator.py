from queue import Queue
import threading
import torch
from typing import List, Tuple
from BRIDGE.generator import Sampler
from BRIDGE.queues import DraftItem, DraftQueue
from BRIDGE.utils.meter import TimeMeter, Statistics
from BRIDGE.utils.mlogging import Logger

logging_level = "INFO"
time_meter = TimeMeter()
stats = Statistics()

BLOCK_DRAFT_TOKEN = 3


class Aggregator(threading.Thread):

    def __init__(
        self,
        draft_queue_loc: DraftQueue,
        draft_queue_rem: DraftQueue,
        target_tokens: Queue,
        sampler: Sampler,
        n_steps: int,
        mode: str,
        target_token_handler: callable = None,
        aggregate_event: threading.Event = None
    ):
        threading.Thread.__init__(self, name=__class__.__name__)
        self.draft_queue_loc = draft_queue_loc
        self.draft_queue_rem = draft_queue_rem
        self.target_tokens = target_tokens
        self.sampler = sampler
        self.logger = Logger.build(__class__.__name__, level=logging_level)
        self.logger.info("Aggregator initialized.")
        self.n_steps = n_steps
        self.step = 0
        self.mode = mode
        self.target_token_handler = target_token_handler
        self.aggregate_event = aggregate_event
        self._stash_loc = []
        self._stash_rem = []
        if mode == "synchronized":
            self.aggregate = self.aggregate_synchronized
        elif mode == "speculative":
            self.aggregate = self.aggregate_speculative
        elif mode == "BRIDGE":
            self.aggregate = self.aggregate_block_speculative
            if self.aggregate_event is None:
                raise ValueError("aggregate_event is required for BRIDGE mode")

    def _get_draft_item(self, queue: DraftQueue) -> DraftItem:
        draft_item = queue.get()
        while draft_item.step != self.step:
            draft_item = queue.get()
        return draft_item

    def _get_draft_items(self, queue: DraftQueue, n_items: int, stash: List[DraftItem]) -> List[DraftItem]:
        draft_items = []
        if stash:
            if stash[0].step != self.step:
                stash.clear()
        while stash and len(draft_items) < n_items:
            draft_items.append(stash.pop(0))
        while len(draft_items) < n_items:
            draft_item = queue.get()
            self.logger.debug(f"Got draft item with step {draft_item.step}, expecting {self.step + len(draft_items)}")
            if draft_item.step < self.step + len(draft_items):
                # Drop stale draft items instead of moving the step backwards.
                self.logger.debug(f"Dropping stale draft item with step {draft_item.step}")
                continue
            while draft_item.step != self.step + len(draft_items):
                self.logger.debug(f"Skipping draft item with step {draft_item.step}, expecting {self.step + len(draft_items)}")
                draft_item = queue.get()
            draft_items.append(draft_item)
        return draft_items

    def run(self):
        self.target_tokens.queue.clear()
        stats.new_record()
        while self.step < self.n_steps:
            if self.mode == "BRIDGE":
                left_tokens = min(BLOCK_DRAFT_TOKEN, self.n_steps - self.step)

                if self._stash_loc and self._stash_loc[0].step != self.step:
                    self._stash_loc.clear()
                if self._stash_rem and self._stash_rem[0].step != self.step:
                    self._stash_rem.clear()

                while (self.draft_queue_loc.qsize() + len(self._stash_loc)) < left_tokens or \
                      (self.draft_queue_rem.qsize() + len(self._stash_rem)) < left_tokens:
                    self.aggregate_event.wait()
                    self.aggregate_event.clear()

                self.logger.debug(f"Start block speculative aggregation for {left_tokens} tokens.")

                draft_loc_items = self._get_draft_items(self.draft_queue_loc, left_tokens, self._stash_loc)
                draft_rem_items = self._get_draft_items(self.draft_queue_rem, left_tokens, self._stash_rem)

                self.logger.debug(f"Draft local len: {len(draft_loc_items)}, Remote len: {len(draft_rem_items)}")

                with time_meter.timer("AggregateLatency"):
                    next_tokens, accept_locs, accept_rems = self.aggregate(draft_loc_items, draft_rem_items)

                timer = time_meter.timer("AggregateLatency")
                if len(next_tokens) > 0:
                    timer.duration /= len(next_tokens)
                stats.update(timer=timer)

                if len(next_tokens) < left_tokens:
                    self._stash_loc = draft_loc_items[len(next_tokens):]
                    self._stash_rem = draft_rem_items[len(next_tokens):]
                self.step += len(next_tokens)
                self.target_token_handler(next_tokens, accept_locs, accept_rems)
            else:
                # synchronized, speculative logic...
                draft_loc = self._get_draft_item(self.draft_queue_loc)
                draft_rem = self._get_draft_item(self.draft_queue_rem)
                with time_meter.timer("AggregateLatency"):
                    next_token, accept_loc, accept_rem = self.aggregate(draft_loc, draft_rem)
                stats.update(time_meter.timer("AggregateLatency"))
                self.target_token_handler(next_token, accept_loc, accept_rem)
                self.step += 1
        self.logger.info("Aggregation complete.")

    def aggregate_synchronized(self, draft_loc: DraftItem, draft_rem: DraftItem):
        device = draft_loc.logprobs.device
        draft_rem.logprobs = draft_rem.logprobs.to(device)
        scores = torch.as_tensor([draft_loc.weight, draft_rem.weight], dtype=torch.float32, device=device)
        scores = torch.log_softmax(scores, dim=0)
        logprobs = torch.stack([draft_loc.logprobs, draft_rem.logprobs], dim=1)  # (s_vocab, 2)
        logprobs = logprobs + scores                             # (s_vocab, 2) + (2,)
        logprobs = torch.logsumexp(logprobs, dim=1)              # (s_vocab,)
        next_token = self.sampler(torch.exp(logprobs))
        return next_token, False, False

    def _speculative_sampling(self, draft_token: int, draft_probs: torch.Tensor, target_probs: torch.Tensor, residual_probs: torch.Tensor):
        if draft_probs[draft_token] <= target_probs[draft_token] \
                or torch.rand(1).to(draft_probs.device) < target_probs[draft_token] / draft_probs[draft_token]:
            return draft_token
        else:
            s = residual_probs.sum()
            if s.item() == 0:
                residual_probs = torch.ones_like(residual_probs) / residual_probs.shape[0]
            token = torch.multinomial(residual_probs, 1).cpu().item()
            return token

    def aggregate_speculative(self, draft_loc: DraftItem, draft_rem: DraftItem):
        device = draft_loc.logprobs.device
        draft_rem.logprobs = draft_rem.logprobs.to(device)
        scores = torch.as_tensor([draft_loc.weight, draft_rem.weight], dtype=torch.float32, device=device)
        scores = torch.log_softmax(scores, dim=0)
        logprobs = torch.stack([draft_loc.logprobs, draft_rem.logprobs], dim=1)  # (s_vocab, 2)
        logprobs = logprobs + scores                             # (s_vocab, 2) + (2,)
        logprobs = torch.logsumexp(logprobs, dim=1)              # (s_vocab,)

        probs_loc = torch.exp(draft_loc.logprobs)
        probs_rem = torch.exp(draft_rem.logprobs)
        probs_agg = self.sampler.transform(torch.exp(logprobs))
        residual_probs_loc = torch.maximum(torch.zeros_like(probs_agg), probs_agg - probs_loc)
        residual_probs_loc = residual_probs_loc / (residual_probs_loc.sum() + 1e-10)
        residual_probs_rem = torch.maximum(torch.zeros_like(probs_agg), probs_agg - probs_rem)
        residual_probs_rem = residual_probs_rem / (residual_probs_rem.sum() + 1e-10)

        next_token_loc = self._speculative_sampling(draft_loc.token, probs_loc, probs_agg, residual_probs_loc)
        next_token_rem = self._speculative_sampling(draft_rem.token, probs_rem, probs_agg, residual_probs_rem)
        next_token = next_token_loc if torch.rand(1) < 0.5 else next_token_rem

        accept_loc = next_token == draft_loc.token
        accept_rem = next_token == draft_rem.token
        return next_token, accept_loc, accept_rem

    def _block_verify(self, draft_tokens: torch.Tensor, draft_probs: torch.Tensor, target_probs: torch.Tensor) -> Tuple[int, int]:
        gamma = draft_tokens.shape[0]
        device = draft_probs.device
        eta_noise = torch.rand(gamma, device=device)
        p_acc = 1.0
        tau = 0

        for i in range(gamma):
            token = draft_tokens[i]
            p_target = target_probs[i, token]
            p_draft = draft_probs[i, token] + 1e-10
            ratio = p_target / p_draft
            p_acc = min(p_acc * ratio.item(), 1.0)
            diff = p_acc * target_probs[i] - draft_probs[i]
            numerator = torch.sum(torch.relu(diff))
            denominator = numerator + (1.0 - p_acc)
            h_block = numerator / (denominator + 1e-10)
            if eta_noise[i] <= h_block:
                tau = i + 1
        return tau

    def dynamic_block_len(self, accept_locs: List[bool], accept_rems: List[bool]) -> int:
        MIN_LEN = 3
        MAX_LEN = 6
        global BLOCK_DRAFT_TOKEN
        if all(accept_locs) or all(accept_rems):
            BLOCK_DRAFT_TOKEN = min(max(BLOCK_DRAFT_TOKEN + 1, MIN_LEN), MAX_LEN)
        else:
            BLOCK_DRAFT_TOKEN = min(max(BLOCK_DRAFT_TOKEN - 1, MIN_LEN), MAX_LEN)

    def aggregate_block_speculative(self, draft_loc_queue: List[DraftItem], draft_remote_queue: List[DraftItem]):
        device = draft_loc_queue[0].logprobs.device

        tokens_L = torch.tensor([item.token for item in draft_loc_queue], device=device)
        probs_L = torch.stack([item.logprobs for item in draft_loc_queue])
        probs_L = probs_L.to(device)
        weights_L = torch.tensor([item.weight for item in draft_loc_queue], device=device)

        tokens_R = torch.tensor([item.token for item in draft_remote_queue], device=device)
        probs_R = torch.stack([item.logprobs for item in draft_remote_queue])
        probs_R = probs_R.to(device)
        weights_R = torch.tensor([item.weight for item in draft_remote_queue], device=device)

        scores = torch.stack([weights_L, weights_R], dim=1).to(probs_L.dtype)
        log_pi = torch.log_softmax(scores, dim=1)
        stacked = torch.stack([probs_L, probs_R], dim=2)
        mix_logprobs = torch.logsumexp(stacked + log_pi.unsqueeze(1), dim=2)
        target_probs = mix_logprobs.exp()

        probs_L_exp = probs_L.exp()
        probs_R_exp = probs_R.exp()
        tau_L = self._block_verify(tokens_L, probs_L_exp, target_probs)
        tau_R = self._block_verify(tokens_R, probs_R_exp, target_probs)

        target_probs = self.sampler.transform(target_probs)
        base_tokens = tokens_L if tau_L >= tau_R else tokens_R
        tau = max(tau_L, tau_R)
        final_tokens = base_tokens[:tau].tolist()

        target_len = len(draft_loc_queue)
        if len(final_tokens) < target_len:
            for i in range(len(final_tokens), target_len):
                dist = target_probs[i] if target_probs.dim() > 1 else target_probs
                final_tokens.append(self.sampler(dist))

        locs_diff = next(
            (i for i, (x, y) in enumerate(zip(tokens_L, final_tokens)) if x != y),
            None,
        )
        rems_diff = next(
            (i for i, (x, y) in enumerate(zip(tokens_R, final_tokens)) if x != y),
            None,
        )
        accept_locs = [True] * len(final_tokens)
        accept_rems = [True] * len(final_tokens)
        if locs_diff is not None:
            accept_locs[locs_diff] = False
        if rems_diff is not None:
            accept_rems[rems_diff] = False
        self.dynamic_block_len(accept_locs, accept_rems)
        return final_tokens, accept_locs, accept_rems
