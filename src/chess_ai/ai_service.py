import threading
from abc import ABC, abstractmethod
from typing import Iterator

from google import genai

from exceptions import APIKeyNotFoundError, AIModelConnectionError, AIModelRequestError, AppError
from logging_config import logger


class AIModelService(ABC):
    """
    Interface to connect to AI models
    """
    @abstractmethod
    def request(self, prompt: str) -> Iterator[str]:
        """
        Method for requesting something from the AI model

        Args:
            prompt (str): instruction to be given to the AI

        Returns:
            Iterator[str]: chunks of the AI response
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Method for closing the connection with the AI model
        """
        pass

    def __enter__(self):
        """
        Enables use of the class as a context manager (with-statement).
        The connection itself is opened lazily on first use, not here.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Ensures the connection is closed when leaving a with-block,
        even if an exception occurred inside it.
        """
        self.close()


class AIGeminiService(AIModelService):
    """
    Class to connect to Gemini AI
    """
    def __init__(self, api_key: str, gemini_version: str) -> None:
        """
        Method for initializing instance attributes

        Args:
            api_key (str): Access key for using Google AI models
            gemini_version (str): Name of the AI model to be used
        """
        self._api_key = api_key
        self.gemini_version = gemini_version
        self.client = None
        self._lock = threading.Lock()

        self.__post_init__()

    def __post_init__(self) -> None:
        """
        Checks if the API key is missing

        Raises:
            APIKeyNotFoundError: if the API key is not found
        """
        if not self._api_key:
            raise APIKeyNotFoundError()

    def _connect(self):
        """
        Initialize and return a connection to the Google Gemini API.

        Returns:
            genai.Client: Initialized Gemini client instance

        Raises:
            AIModelConnectionError: if the connection to Gemini fails
        """
        logger.info("Opening connection with Gemini model")

        try:
            client = genai.Client(api_key=self._api_key)
        except Exception as e:
            raise AIModelConnectionError(str(e)) from e

        return client

    def request(self, prompt: str) -> Iterator[str]:
        """
        Method for requesting something from the AI model.

        Opens a connection immediately if one doesn't exist yet
        (protected by a lock against concurrent calls), then returns
        a generator that streams the response.

        Args:
            prompt (str): The input prompt/message for the model

        Returns:
            Iterator[str]: Text chunks from the model response

        Raises:
            AIModelConnectionError: if opening the connection fails
        """
        with self._lock:
            if not self.client:
                self.client = self._connect()

        return self._stream_response(prompt)

    def _stream_response(self, prompt: str) -> Iterator[str]:
        """
        Generator that yields chunks of text from an already-open connection.

        Args:
            prompt (str): The input prompt/message for the model

        Yields:
            str: Text chunks from the model response

        Raises:
            AIModelRequestError: if the request or streaming fails
        """
        try:
            stream = self.client.models.generate_content_stream(
                model=self.gemini_version,
                contents=prompt
            )

            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except AppError:
            raise
        except Exception as e:
            raise AIModelRequestError(str(e)) from e

    def close(self) -> None:
        """
        Method for closing the connection with the AI model.

        Protected by a lock to avoid closing a client that another
        thread is in the middle of creating.

        Raises:
            AIModelConnectionError: if closing the connection fails
        """
        with self._lock:
            if self.client:
                logger.info("Closing connection with Gemini model")
                try:
                    self.client.close()
                except Exception as e:
                    raise AIModelConnectionError(str(e)) from e
                finally:
                    self.client = None
