from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def simulate(self, *args, **kwargs):
        pass

