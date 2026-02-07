
from queue import Queue
import threading
from BRIDGE.utils.meter import Statistics, TimeMeter
from .rag import Rag
from .generator import CausalOutput
from .queues import DraftItem
from .utils.mlogging import Logger
from .aggregator import BLOCK_DRAFT_TOKEN


logging_level = "INFO"
time_meter = TimeMeter()
stats = Statistics()


class Decoder(threading.Thread):

    def __init__(self, rag: Rag, draft_token_handler: callable, output_tokens: Queue, query: str, prompt_template: str, n_steps: int, aggregation_mode: str, add_to_draft_queue_func: callable, scroll_back_pos: list, decode_event: threading.Event):
        threading.Thread.__init__(self, name=__class__.__name__)
        self.logger = Logger.build(__class__.__name__, logging_level)
        self.rag = rag
        self.draft_token_handler = draft_token_handler
        self.query, self.template, self.n_steps = query, prompt_template, n_steps
        self.output_tokens = output_tokens
        self.aggregation_mode = aggregation_mode
        self.draft_queue: Queue[DraftItem] = Queue(0)
        self.add_to_draft_queue_func: callable = add_to_draft_queue_func
        self.scroll_back_pos = scroll_back_pos
        self.logger.info("Decoder initialized.")

        self.input_ids, self.attention_mask, self.scores, self.passages = self.rag._prepare_inputs_for_generation(self.query, self.template)
        self.context_length = self.input_ids.shape[1]
        self.output_tokens.queue.clear()
        self.rag.generator.preempt_event.clear()

        self.decode_event = decode_event

        self.output_ids = []
        self.step = 0

        global stats
        stats.new_record()

    def prefilling(self) -> CausalOutput:
        output = self.rag._generate(self.input_ids, self.attention_mask, self.scores)
        draft_item = DraftItem(
            token=output.next_token, logprobs=output.logprobs,
            weight=output.weight, step=self.step
        )
        self.add_to_draft_queue_func(draft_item)
        self.draft_queue.put(draft_item)
        if self.aggregation_mode == "BRIDGE":
            self._synchronize_batch_output_to_remote()
        else:
            self._synchronize_output_to_remote(output)
        # self.step += 1
        return output

    def _scroll_back(self, output: CausalOutput) -> CausalOutput:
        # scroll back step
        # Remove trailing sentinel token(s) (-1) that indicate rejected block-spec steps.
        # with self.output_tokens.mutex:
        #     while self.output_tokens.queue and self.output_tokens.queue[-1] == -1:
        #         self.logger.debug("Dropping sentinel token -1 from output queue before scroll back.")
        #         self.output_tokens.queue.pop()
        #     qsize = len(self.output_tokens.queue)
        qsize = self.scroll_back_pos[0]

        # After removing sentinels, re-compute the current step.
        # Align with normal flow where prefilling accounts for step 0.
        self.step = max(qsize - 1, 0)
        self.logger.debug(f"Scrolling back to step {self.step}.")

        # scroll back input_ids (if any token remains)
        if qsize > 0:
            output.next_token = self.output_tokens.queue[-1]

        # scroll back attention_mask
        cur_seq_len = self.context_length + self.step
        self.attention_mask = self.attention_mask[:, : cur_seq_len]

        # TODO: scroll back the scores
        # output.weight = ...

        # scroll back key valuse cache
        output.past_key_values = list(output.past_key_values)
        for i, _ in enumerate(output.past_key_values):
            output.past_key_values[i] = list(output.past_key_values[i])
            output.past_key_values[i][0] = output.past_key_values[i][0][..., : cur_seq_len, :]
            output.past_key_values[i][1] = output.past_key_values[i][1][..., : cur_seq_len, :]
            output.past_key_values[i] = tuple(output.past_key_values[i])
        output.past_key_values = tuple(output.past_key_values)

        self.draft_queue = Queue(0)
        return output

    def _synchronize_output_to_remote(self, output: CausalOutput):
        if self.step > self.n_steps:
            return
        draft_item = DraftItem(
            token=output.next_token, logprobs=output.logprobs,
            weight=output.weight, step=self.step
        )
        if self.aggregation_mode == "BRIDGE":
            draft_item = [draft_item]
        self.draft_token_handler(draft_item)

    def _synchronize_batch_output_to_remote(self, force=False):
        if self.step > self.n_steps:
            return
        if not force:
            batch_size = min(BLOCK_DRAFT_TOKEN, self.n_steps - self.output_tokens.qsize())
            # print("batch size:", batch_size)
            if self.draft_queue.qsize() < batch_size or self.draft_queue.qsize() == 0:
                return
        else:
            batch_size = self.draft_queue.qsize()
            if batch_size == 0:
                return
        draft_items = [self.draft_queue.get() for _ in range(batch_size)]
        self.draft_token_handler(draft_items)

    def decoding(self, output: CausalOutput):
        while self.output_tokens.qsize() < self.n_steps:
            if self.step >= self.n_steps:
                if self.aggregation_mode == "BRIDGE":
                    self._synchronize_batch_output_to_remote(force=True)
                # Avoid deadlock when no preempt occurs; re-check output size periodically.
                self.decode_event.wait(timeout=0.05)
                self.decode_event.clear()
                if self.output_tokens.qsize() >= self.n_steps:
                    break
                continue
            self.logger.debug(f"step {self.step}: n_output_tokens={self.output_tokens.qsize()}")
            with time_meter.timer("latency_dec_loc"):
                # self.logger.debug(f"next_token before generation: {output.next_token}")
                temp_output, temp_attention_mask = self.rag.generate(
                    output.next_token, self.scores, self.attention_mask, past_key_values=output.past_key_values)
            if not temp_output:
                output = self._scroll_back(output)
            else:
                stats.update(time_meter.timer("latency_dec_loc"))
                stats.update(name="steps", stat=self.step)
                step_len = len(output.next_token) if isinstance(output.next_token, list) else 1
                output, self.attention_mask = temp_output, temp_attention_mask
                self.step += step_len
                draft_item = DraftItem(
                    token=output.next_token, logprobs=output.logprobs,
                    weight=output.weight, step=self.step
                )

                self.add_to_draft_queue_func(draft_item)

                if self.aggregation_mode == "BRIDGE":
                    self.draft_queue.put(draft_item)
                    self._synchronize_batch_output_to_remote()
                else:
                    self._synchronize_output_to_remote(output)

    def run(self):
        self.decoding(self.prefilling())
        self.rag.generator.preempt_event.set()
        self.output_ids = [self.output_tokens.get() for _ in range(self.n_steps)]
        self.logger.debug("Generation complete.")
