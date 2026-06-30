from abc import ABC, abstractmethod
from typing import Iterator


class AIModelService(ABC):
    @abstractmethod
    def request(self, prompt: str) -> Iterator[str]:
        pass


class AIGeminiService(AIModelService):
    def __init__(self) -> None:
        pass
    
    def _connect(self):
        pass

    def request(self, prompt: str) -> Iterator[str]:
        pass


def main():
    pass


if __name__ == "__main__":
    main()