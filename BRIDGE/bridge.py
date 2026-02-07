import threading
from queue import Queue
from typing import List
from tqdm import tqdm
from .rag import Rag
from .transceiver import Message
from .config import BRIDGEConfig
from .aggregator import Aggregator
from .decoder import Decoder
from .queues import DraftQueue, DraftItem
from .transceiver import Transceiver
from .utils.stable import terminate_thread
from .utils.mlogging import Logger
from .utils.meter import Statistics, TimeMeter
from .aggregator import BLOCK_DRAFT_TOKEN

logging_level = "INFO"
time_meter = TimeMeter()


class BRIDGE:

    def __init__(self, config: BRIDGEConfig):
        self.logger = Logger.build(__class__.__name__, level=logging_level)
        self.ready_for_generation = False
        self.config = config
        self.process_bar = None
        self.stats = Statistics()
        self.is_client = config.trans.rank != 0
        self.aggregation_mode = config.aggregator.mode

        self.draft_queue_rem = DraftQueue()
        self.draft_queue_loc = DraftQueue()
        self.aggregate_event = threading.Event()

        self.target_tokens = Queue(0)
        self.send_target_tokens_queue: Queue[DraftItem] = Queue(0)
        self.output_tokens = Queue(0)

        self.aggregator = None
        self.decoder = None
        self.scroll_back_pos = [0, ]
        self.current_latency = None

        self.rag = Rag(config)
        self.transceiver = Transceiver(config)
        self.transceiver.register_observers(self._collect_observers())
        self.transceiver.send(Message.READY_FOR_GENERATION, None)

    def shutdown(self):
        self.transceiver.send(Message.SHUTDOWN, None)
        self._shutdown()

    def _shutdown(self):
        self.logger.info("Shutting down.")
        terminate_thread(self.aggregator)
        terminate_thread(self.decoder)
        terminate_thread(self.rag.generator)
        self.logger.info("BRIDGE threads shutdown.")
        self.transceiver.terminate()

    def _build_aggregator(self, max_new_tokens: int):
        thread = Aggregator(
            self.draft_queue_loc,
            self.draft_queue_rem,
            self.target_tokens,
            self.rag.generator.sampler,
            max_new_tokens,
            self.aggregation_mode,
            (self._send_batch_target_tokens
             if self.aggregation_mode == "BRIDGE"
             else self._send_target_token),
            self.aggregate_event
        )
        thread.start()
        return thread

    def _build_decoder(self, query, prompt_template, max_new_tokens):
        _send_draft_method = (self._send_draft_token
                              if self.aggregation_mode != "BRIDGE"
                              else self._send_batch_draft_tokens)
        self.decode_event = threading.Event()
        thread = Decoder(
            self.rag, _send_draft_method, self.output_tokens,
            query, prompt_template, max_new_tokens, self.aggregation_mode,
            self._put_local_draft_item, self.scroll_back_pos,
            self.decode_event)
        thread.start()
        return thread

    def _put_local_draft_item(self, draft_item: DraftItem):
        # if self.is_client:
        #     time.sleep(0.2)
        self.draft_queue_loc.put(draft_item)
        self.aggregate_event.set()

    def _clean_up(self):
        self.transceiver.receive_queue.queue.clear()
        self.rag.generator.input_queue.queue.clear()
        self.rag.generator.output_queue.queue.clear()
        self.draft_queue_loc.clear()
        self.draft_queue_rem.clear()
        self.decoder.draft_queue = Queue(0)
        self.logger.debug("Cleaned up.")

    def _start_up(self, query, prompt_template, max_new_tokens):
        self.stats.new_record()
        self.process_bar = tqdm(total=max_new_tokens, desc="Generating", leave=False)
        self.recompute_checker = threading.Thread(
            target=self.check_recompute, args=(max_new_tokens,))
        self.recompute_checker.start()
        self.decoder = self._build_decoder(query, prompt_template, max_new_tokens)
        if self.is_client:
            self.aggregator = self._build_aggregator(max_new_tokens)
            self.decoder.join()
            self.aggregator.join()
            self.recompute_checker.join()
            self._clean_up()

    def query(self, query: str, prompt_template: str, max_new_tokens: int, local_context: str = None):
        self.max_new_tokens = max_new_tokens
        self.rag.local_context = local_context
        self._send_begin_generate(query, prompt_template, max_new_tokens)
        if local_context is not None:
            query = local_context
        self._start_up(query, prompt_template, max_new_tokens)

        # Get output text
        output_ids = self.decoder.output_ids
        output_txt = self.rag.generator.tokenizer.decode(
            output_ids, skip_special_tokens=True)
        self.process_bar.close()

        self.rag.local_context = None
        return output_txt

    def check_recompute(self, n_steps: int):
        step = 0
        last_block_len = BLOCK_DRAFT_TOKEN
        while step < n_steps:
            with time_meter.timer("LatencyBatchToken"):
                target_tokens, accept_locs, accept_rems = self.target_tokens.get()
                self.logger.debug(f"CheckRecompute received target tokens: {target_tokens}")

            self.stats.update(time_meter.timer('LatencyBatchToken'))
            self.stats.update(name="NumTokensPerBatch", stat=len(target_tokens) if isinstance(target_tokens, list) else 1)

            if isinstance(accept_locs, bool):
                accept_locs = [accept_locs]
            if isinstance(accept_rems, bool):
                accept_rems = [accept_rems]
            if isinstance(target_tokens, int):
                target_tokens = [target_tokens]

            for accept_loc, accept_rem in zip(accept_locs, accept_rems):
                self.stats.update(name="AcceptanceLoc", stat=accept_loc)
                self.stats.update(name="AcceptanceRem", stat=accept_rem)

            reject_idx = None
            for i, (accept_loc, accept_rem) in enumerate(zip(accept_locs, accept_rems)):
                if not accept_loc or not accept_rem:
                    reject_idx = i
                    break

            step_len = 0
            for token in target_tokens:
                if token != -1:
                    if self.output_tokens.qsize() >= n_steps:
                        break
                    self.output_tokens.put(token)
                    step_len += 1
            if reject_idx is not None:
                self.scroll_back_pos[0] = self.output_tokens.qsize()
                if self.aggregation_mode == "BRIDGE" and self.aggregator is not None:
                    self.aggregator.step = self.scroll_back_pos[0]
                    self.aggregator._stash_loc.clear()
                    self.aggregator._stash_rem.clear()
                self.rag.generator.preempt_event.set()
                self.decode_event.set()

                self.draft_queue_loc.discard_before(self.scroll_back_pos[0])
                self.draft_queue_rem.discard_before(self.scroll_back_pos[0])

            if self.aggregation_mode == "BRIDGE" and self.is_client:
                if last_block_len != BLOCK_DRAFT_TOKEN:
                    self._send_block_len(BLOCK_DRAFT_TOKEN)
                    last_block_len = BLOCK_DRAFT_TOKEN
            step += step_len
            self.process_bar.update(step_len)
            self.logger.debug(f"Generation progress: {step}/{n_steps} tokens.")
        self.logger.debug("Recompute checker finished processing all steps.")

    def _collect_observers(self):
        return [
            self._rx_ready_for_generation,
            self._rx_begin_generate,
            self._rx_draft_token,
            self._rx_target_token,
            self._rx_shutdown,
            self._rx_batch_draft_tokens,
            self._rx_batch_target_tokens
        ]

    def _rx_ready_for_generation(self, mtype: int, mbody: object):
        if mtype != Message.READY_FOR_GENERATION:
            return False
        self.ready_for_generation = True
        self.logger.debug("Remote is ready for generation.")
        return True

    def _rx_begin_generate(self, mtype: int, mbody: object):
        if mtype != Message.BEGIN_GENERATE:
            return False
        query, prompt_template, max_new_tokens = mbody
        self.logger.debug(f"Generating response for query: {query}")
        # self.profiler = self._build_profiler(query, prompt_template, max_new_tokens)
        # self._send_profile(self.profiler.stats.records)
        self._start_up(query, prompt_template, max_new_tokens)
        return True

    def _rx_draft_token(self, mtype: int, mbody: object):
        if mtype != Message.DRAFT_TOKEN:
            return False
        self.draft_queue_rem.put(DraftItem.from_tuple(mbody))
        self.aggregate_event.set()
        return True

    def _rx_target_token(self, mtype: int, mbody: object):
        if mtype != Message.TARGET_TOKEN:
            return False
        self.target_tokens.put(mbody)
        return True

    def _rx_shutdown(self, mtype: int, mbody: object):
        if mtype != Message.SHUTDOWN:
            return False
        self.logger.info("Received shutdown signal.")
        self._shutdown()
        return True

    def _rx_batch_draft_tokens(self, mtype: int, mbody: object):
        if mtype != Message.BATCH_DRAFT_TOKENS:
            return False
        draft_items = [DraftItem.from_tuple(tup) for tup in mbody]
        self.draft_queue_rem.put_many(draft_items)
        self.aggregate_event.set()
        return True

    def _rx_batch_target_tokens(self, mtype: int, mbody: object):
        if mtype != Message.BATCH_TARGET_TOKENS:
            return False
        tokens, accept_rems, accept_locs = mbody
        self.target_tokens.put((tokens, accept_rems, accept_locs))
        return True

    def rx_block_len(self, mtype: int, mbody: object):
        if mtype != Message.BLOCK_LEN:
            return False
        block_len = mbody
        global BLOCK_DRAFT_TOKEN
        BLOCK_DRAFT_TOKEN = block_len
        return True

    def _send_begin_generate(self, query: str, prompt_template: str, max_new_tokens: int):
        payload = (query, prompt_template, max_new_tokens)
        self.transceiver.send(Message.BEGIN_GENERATE, payload)

    def _send_draft_token(self, draft_item: DraftItem):
        self.transceiver.send(Message.DRAFT_TOKEN, draft_item.as_tuple())

    def _send_batch_draft_tokens(self, draft_items: List[DraftItem]):
        tuples = [item.as_tuple() for item in draft_items]
        self.transceiver.send(Message.BATCH_DRAFT_TOKENS, tuples)

    def _send_target_token(self, token: int, accept_loc: bool, accept_rem: bool):
        self.target_tokens.put((token, accept_loc, accept_rem))
        self.transceiver.send(Message.TARGET_TOKEN, (token, accept_rem, accept_loc))

    def _send_batch_target_tokens(self, tokens: List[int], accept_locs: List[bool], accept_rems: List[bool]):
        if self.output_tokens.qsize() >= self.max_new_tokens:
            return
        if len(tokens) == 0:
            return
        self.target_tokens.put((tokens, accept_locs, accept_rems))
        self.transceiver.send(Message.BATCH_TARGET_TOKENS, (tokens, accept_rems, accept_locs))

    def _send_block_len(self, block_len: int):
        self.transceiver.send(Message.BLOCK_LEN, block_len)
