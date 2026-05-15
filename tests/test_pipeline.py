import queue
import time

from poker_vision.core.pipeline import Orchestrator, Stage


class ProducerStage(Stage):
    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def run(self):
        for i in range(1, self.limit + 1):
            # 1. Verifica se alguém apertou o "Ctrl+C" antes de criar o próximo item
            if self.stop_event and self.stop_event.is_set():
                break

            # 2. Tenta colocar o item na fila de forma segura (lidando com a fila cheia)
            while not (self.stop_event and self.stop_event.is_set()):
                try:
                    self.out_q.put(i, timeout=0.1)
                    break
                except queue.Full:
                    # A fila encheu (Backpressure agindo).
                    # O loop repete e verifica o stop_event novamente.
                    pass

    def process(self, item):
        pass


class DoublerStage(Stage):
    def process(self, item: int) -> int:
        return item * 2


class PrinterStage(Stage):
    def __init__(self):
        super().__init__()
        self.results = []

    def process(self, item: int) -> None:
        self.results.append(item)
        print(f"Resultado final: {item}")
        return None


def test_pipeline_chaining_and_clean_shutdown():
    producer = ProducerStage(limit=3)
    doubler = DoublerStage()
    printer = PrinterStage()

    orchestrator = Orchestrator([producer, doubler, printer])
    orchestrator.start()

    time.sleep(0.5)

    orchestrator.stop()

    # Verifica se os três estágios morreram
    assert not producer.is_alive()
    assert not doubler.is_alive()
    assert not printer.is_alive()

    assert printer.results == [2, 4, 6]


def test_long_execution_ctrl_c_simulation():
    class SlowStage(Stage):
        def process(self, item):
            time.sleep(0.1)  # Simula trabalho pesado (como o YOLO)
            return item

    p1 = ProducerStage(limit=100)  # Produz 100 itens (vai demorar 10 segundos)
    p2 = SlowStage()
    p3 = PrinterStage()

    orchestrator = Orchestrator([p1, p2, p3])
    orchestrator.start()

    # Simula o usuário apertando Ctrl+C no meio da execução (apos 0.5s)
    time.sleep(0.5)

    start_time = time.time()
    orchestrator.stop(timeout=2.0)
    end_time = time.time()

    # Verifica se fechou dentro de 2 segundos (DoD 2)
    assert (end_time - start_time) < 2.0
    assert not p1.is_alive()
    assert not p2.is_alive()
    assert not p3.is_alive()
