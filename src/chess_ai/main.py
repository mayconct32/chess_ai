from abc import ABC, abstractmethod
from typing import Iterator
from os import getenv
import logging

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs.log', encoding='utf-8')


class APIKeyNotFoundError(Exception):
    pass


class AIModelService(ABC):
    @abstractmethod
    def request(self, prompt: str) -> Iterator[str]:
        pass


class AIGeminiService(AIModelService):
    def __init__(self, api_key: str, gemini_version: str) -> None:
        self._api_key = api_key
        if not self._api_key:
            raise APIKeyNotFoundError(
                "API_KEY not found. Please set it in your .env file."
                )
        self.gemini_version = gemini_version
        self.connection = None

    def _connect(self) -> ChatGoogleGenerativeAI:
        model = ChatGoogleGenerativeAI(
            model=self.gemini_version,
            google_api_key=self._api_key
        )
        return model

    def request(self, prompt: str) -> Iterator[str]:
        if not self.connection:
            self.connection = self._connect()
        for chunk in self.connection.stream(prompt):
            if chunk.content:
                yield chunk.content[0]["text"]


def main():
    try:
        API_KEY =  getenv("API_KEY")
        GEMINI_VERSION = "gemini-3.1-flash-lite"
        gemini_service = AIGeminiService(API_KEY, GEMINI_VERSION)
        for chunk in gemini_service.request("olá, Chat. Meu nome é magal."):
            print(chunk, end="")
        print()
    except APIKeyNotFoundError as e:
        logger.exception(e)
        print(f"\033[31m APIKeyNotFoundError: {e}\033[m")
    except Exception as e:
        logger.exception(e)
        print(f"\033[31m Internal server error: {e}\033[m")


if __name__ == "__main__":
    main()