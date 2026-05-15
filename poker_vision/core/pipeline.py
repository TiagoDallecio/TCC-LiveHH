import queue
import threading
from abc import ABC, abstractmethod
from typing import Any, List, Optional

# TODO
"""
NOTA: A solução de processamento atual de filas não é a mais performática e pode
causar travamentos durante a análise de um 'POV' ao vivo. Mais para frente será implementada
a melhoria de utilizar 'Frame Dropping no lugar do Backpressure.
"""


class ShutdownSignal:
    pass


SHUTDOWN = ShutdownSignal()


class Stage(threading.Thread, ABC):
    """
    Estágio abstrato baseado em thread.
    Lê de in_q, chama process(), e escreve em out_q.
    """

    def __init__(self, max_q_size: int = 30) -> None:
        super().__init__(name=self.__class__.__name__)
        self.in_q: queue.Queue = queue.Queue(maxsize=max_q_size)
        self.out_q: Optional[queue.Queue] = None
        self.stop_event: Optional[threading.Event] = None
        self.daemon = True  # Permite que o programa feche se a thread travar

    @abstractmethod
    def process(self, item: Any) -> Any:
        """Processa um item da fila. Deve ser implementado nas subclasses."""
        pass

    def run(self) -> None:
        while not (self.stop_event and self.stop_event.is_set()):
            try:
                item = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Se for o sinal de desligamento, repassa para o próximo e morre
            if item is SHUTDOWN:
                break

            try:
                result = self.process(item)
                # Só passa para o próximo estágio se não for None
                if result is not None and self.out_q is not None:
                    while not (self.stop_event and self.stop_event.is_set()):
                        try:
                            self.out_q.put(result, timeout=0.1)
                            break
                        except queue.Full:
                            pass
            except Exception as e:
                print(f"[Erro no {self.name}]: {e}")
            finally:
                self.in_q.task_done()


class Orchestrator:
    """
    Interliga as filas dos estágios, liga as threads e desliga graciosamente.
    """

    def __init__(self, stages: List[Stage]) -> None:
        if not stages:
            raise ValueError("O Orchestrator precisa de pelo menos 1 estágio.")

        self.stages = stages
        self.stop_event = threading.Event()

        # Interligando os estágios (O output de um é o input do outro)
        for i in range(len(self.stages) - 1):
            self.stages[i].out_q = self.stages[i + 1].in_q
            self.stages[i].stop_event = self.stop_event

        self.stages[-1].out_q = None
        self.stages[-1].stop_event = self.stop_event

    def start(self) -> None:
        """Inicia todas as threads."""
        for stage in self.stages:
            stage.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Injeta o sentinela e dá join em todas as threads."""
        print("\nDesligando pipeline...")
        self.stop_event.set()

        # Injeta o shutdown no primeiro estágio e propaga para os próximos
        self.stages[0].in_q.put(SHUTDOWN)

        for stage in self.stages:
            stage.join(timeout=timeout)
            if stage.is_alive():
                print(f"Aviso: {stage.name} não encerrou dentro de {timeout}s.")
        print("Pipeline encerrado.")
