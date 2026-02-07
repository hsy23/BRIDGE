import queue
import struct
import pickle
import time
import threading
import socket

from typing import List, Protocol, Tuple
from logging import Logger as PyLogger
from .config import BRIDGEConfig
from .utils.mlogging import Logger
from .utils.stable import terminate_thread
import numpy as np
from BRIDGE.utils.meter import TimeMeter, Statistics
time_meter = TimeMeter()
stats = Statistics()

logging_level = "INFO"


class Message:
    ACK = 0
    READY_FOR_CONNECTION = 1
    READY_FOR_GENERATION = 2
    BEGIN_GENERATE = 3
    DRAFT_TOKEN = 4
    TARGET_TOKEN = 5
    SHUTDOWN = 6
    EMPTY = 7
    DRAT_SEQUENCE = 8
    PREPARE_COMPLETE = 9
    BEGIN_DECODE = 10
    KV_CACHE = 11
    PROFILE = 12
    BATCH_DRAFT_TOKENS = 13
    BATCH_TARGET_TOKENS = 14
    BLOCK_LEN = 15
    header = ">B I"

    @staticmethod
    def unpack(data: bytes) -> object:
        return pickle.loads(data)

    @staticmethod
    def pack(mtype: int, mbody: object) -> Tuple[bytes, int]:
        if not 0 <= mtype <= 255:
            raise ValueError("mtype should be in range 0-255")
        mbody = pickle.dumps(mbody)
        data = struct.pack(Message.header, mtype, len(mbody)) + mbody
        return data, len(mbody)

    @staticmethod
    def empty_message():
        return Message.EMPTY, pickle.dumps(None)


mtype2str = {getattr(Message, attr): attr for attr in dir(Message) if not attr.startswith("_")}

LATENCY = 0
JITTER = LATENCY / 5
RX_BANDWIDTH = 0  # bytes per second
START_TIME = time.perf_counter()


class NetworkMeter(TimeMeter):
    class NetworkTimer(TimeMeter.Timer):
        def __enter__(self):
            global stats
            super().__enter__()
            byte_size = self.context.get('byte_size', 0)
            # stats.new_record()
            stats.update(name="DataSize", stat=byte_size)
            return self

        def __exit__(self, *exc):
            global stats
            super().__exit__(*exc)
            stats.update(name="NetworkLatency", stat=self.duration)

    def timer(self, name, **context) -> 'NetworkMeter.NetworkTimer':
        return self.NetworkTimer(name, **context)


network_meter = NetworkMeter()


def latency_trace(t: float):
    np.random.seed(int(t) * 100)
    return 0.002 + np.random.uniform(0, 0.006) + LATENCY/1000.0 + np.sin(t / 10) * JITTER/1000.0


class ReceiveListener(threading.Thread):
    def __init__(
            self, socket: socket.socket,
            queue: queue.Queue,
            logger: PyLogger,
            rx_latency_ms: int = LATENCY,
            rx_jitter_ms: int = JITTER):
        threading.Thread.__init__(self, name=__class__.__name__)
        self.stop_event = threading.Event()
        self.queue = queue
        self.socket = socket
        self.socket.listen(1)
        self.logger = logger
        self.header_size = struct.calcsize(Message.header)
        self.conn = None
        # Network simulation knobs (RX side)
        # self.rx_latency_ms = max(0, int(rx_latency_ms or 0))
        # self.rx_jitter_ms = max(0, int(rx_jitter_ms or 0))
        self.logger.info(f"Protocol header size: {self.header_size} Bytes")

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def run(self):
        self.conn, addr = self.socket.accept()
        self.logger.info(f"Connection accepted from {addr}")
        while not self.stop_event.is_set():
            try:
                header = self.conn.recv(self.header_size, socket.MSG_WAITALL)
                if not header or len(header) < self.header_size:
                    self.logger.debug("Connection closed while receiving header.")
                    break
                mtype, body_len = struct.unpack(Message.header, header)
                if mtype == Message.DRAT_SEQUENCE:
                    stats.new_record()
                    with time_meter.timer("ReceiveDraftSequenceLatency"):
                        mbody = self.conn.recv(body_len, socket.MSG_WAITALL)
                    stats.update(time_meter.timer("ReceiveDraftSequenceLatency"))
                    stats.update(name="DraftSequenceSize", stat=body_len)
                else:
                    mbody = self.conn.recv(body_len, socket.MSG_WAITALL)
            except Exception as e:
                self.logger.debug(f"Exception occurred while receiving data: {e}")
                import traceback
                traceback.print_exc()
                break
            # # Simulate RX latency
            # if self.rx_latency_ms > 0 or self.rx_jitter_ms > 0:
            #     base = self.rx_latency_ms / 1000.0
            #     jitter = random.uniform(0, self.rx_jitter_ms) / 1000.0 if self.rx_jitter_ms > 0 else 0.0
            #     time.sleep(base + jitter)
            self.queue.put((mtype, mbody))
            if mtype in [Message.DRAFT_TOKEN, Message.TARGET_TOKEN]:
                self.logger.debug(f"Received Message(mtype={mtype2str[mtype]}, len={body_len}, token={pickle.loads(mbody)[0]})")
            else:
                self.logger.debug(f"Received Message(mtype={mtype2str[mtype]}, len={body_len})")
        self.logger.debug("Connection closed.")
        if self.conn:
            self.conn.close()


class Observer(Protocol):
    def __call__(self, mtype: int, mbody: object) -> bool:
        ...


class ReceiveHandler(threading.Thread):
    def __init__(
            self, queue: queue.Queue,
            observers: List[Observer]):
        threading.Thread.__init__(self, name=__class__.__name__)
        self.stop_event = threading.Event()
        self.queue = queue
        self.observers = observers
        # self.logger = Logger.build(__class__.__name__, level="INFO")

    def close(self):
        self.stop_event.set()
        self.queue.put(Message.empty_message())

    def run(self):
        while not self.stop_event.is_set():
            mtype, mbody = self.queue.get()
            mbody = Message.unpack(mbody)
            # self.logger.info(f"Unpacked Message(mtype={mtype2str[mtype]})")
            self.notify(mtype, mbody)

    def notify(self, mtype: int, mbody: object):
        for observer in self.observers:
            observer(mtype, mbody)
            # if notified:
            #     self.logger.info(f"Notified observer `{observer.__name__}`.")


class Transceiver:

    def __init__(self, config: BRIDGEConfig):
        self.logger = Logger.build(__class__.__name__, level=logging_level, format="[bold yellow1][Device]:Transceiver:[/bold yellow1] %(message)s")
        self.config = config
        self.config.trans.rx_latency_ms = LATENCY
        self.config.trans.rx_jitter_ms = JITTER

        self.remote_ip = config.trans.tx_host
        self.local_ip = config.trans.rx_host
        self.init_receiver(port=self.config.trans.rx_port)
        self.logger.info("Receiver initialized.")
        self.init_sender(port=self.config.trans.tx_port)
        self.logger.info("Sender initialized.")

    def init_receiver(self, port):
        self.logger.debug(f"Binding to local address {self.local_ip}:{port}...")
        self.rx_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rx_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        while True:
            try:
                self.rx_socket.bind((self.local_ip, port))
                break
            except OSError:
                time.sleep(3)

        self.receive_queue = queue.Queue(0)
        self.receive_listener = ReceiveListener(
            self.rx_socket,
            self.receive_queue,
            self.logger,
            rx_latency_ms=getattr(self.config.trans, 'rx_latency_ms', 0),
            rx_jitter_ms=getattr(self.config.trans, 'rx_jitter_ms', 0))
        self.receive_listener.start()

    def init_sender(self, port):
        self.tx_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Store TX-side network simulation knobs
        self.tx_latency_ms = max(0, int(getattr(self.config.trans, 'tx_latency_ms', 0) or 0))
        self.tx_jitter_ms = max(0, int(getattr(self.config.trans, 'tx_jitter_ms', 0) or 0))
        while True:
            try:
                self.logger.info("Trying to connect to the remote node...")
                self.tx_socket.connect((self.remote_ip, port))
                break
            except ConnectionRefusedError:
                self.logger.debug(f"Connection refused by {self.remote_ip}:{port}, retrying...")
                time.sleep(3)
            except OSError as e:
                self.logger.error(e)
                self.tx_socket.close()
                self.tx_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                time.sleep(3)
        self.logger.info(f"Connected to remote node at {self.remote_ip}:{port}.")
        self.send_queue = queue.Queue(0)
        self.send_thread = threading.Thread(
            target=self._send_thread, name="SendThread")
        self.send_thread.start()

    def send(self, mtype: int, mbody: object):
        self.send_queue.put((mtype, mbody))

    def simulate_tx_latency(self):
        if LATENCY > 0:
            time.sleep(latency_trace(time.perf_counter()))

    def _send_thread(self):
        while True:
            mtype, mbody = self.send_queue.get()
            if mtype == Message.EMPTY:
                break
            data, body_len = Message.pack(mtype, mbody)
            try:
                with network_meter.timer("TransmitLatency", byte_size=len(data)):
                    self.simulate_tx_latency()
                    self.tx_socket.sendall(data)
            except Exception as e:
                self.logger.debug(f"Failed to send data: {e}")
                break
            if mtype in [Message.DRAFT_TOKEN, Message.TARGET_TOKEN]:
                self.logger.debug(f"Sent Message(mtype={mtype2str[mtype]}, len={body_len}, token={mbody[0]})")
            else:
                self.logger.debug(f"Sent Message(mtype={mtype2str[mtype]}, len={body_len})")

    def register_observers(self, observers: List[Observer]):
        self.receive_handler = ReceiveHandler(
            self.receive_queue, observers)
        self.receive_handler.start()

    def terminate(self):
        self.rx_socket.close()
        self.tx_socket.close()
        self.logger.info("Socket closed.")
        self.send_queue.put(Message.empty_message())
        self.send_thread.join()
        terminate_thread(self.receive_listener)
        terminate_thread(self.receive_handler)
        self.logger.info("Transceiver stopped.")
