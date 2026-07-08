from abc import ABC, abstractmethod
from typing import Iterator, List
import os
import logging
import threading

from dotenv import load_dotenv
from google import genai
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma.vectorstores import Chroma


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500
API_KEY = os.getenv("API_KEY")
GEMINI_VERSION = "gemini-3.1-flash-lite"
DATABASE_DIR = "database"


class AppError(Exception):
    """
    Base class for every custom exception in this project.

    Logs the message a single time, at creation, so callers never need
    to write "logger.error(...)" right before a "raise" — that pattern
    just repeats the exception's own message as boilerplate. Any
    exception here should inherit from this class instead of Exception.
    """
    def __init__(self, message: str) -> None:
        """
        Method for initializing and logging the exception message

        Args:
            message (str): Human-readable description of the error
        """
        logger.error(message)
        super().__init__(message)


class APIKeyNotFoundError(AppError):
    """
    Custom exception for API key not found
    """
    def __init__(self):
        """
        Method for initializing the exception message
        """
        super().__init__("API_KEY not found. Please set it in your .env file.")


class PDFFileNotFoundError(AppError):
    """
    Custom exception for file not found
    """
    def __init__(self, file_path):
        """
        Method for initializing the exception message

        Args:
            file_path (str): Path of the PDF file that was not found
        """
        super().__init__(f"PDF file not found: {file_path}")


class AIModelError(AppError):
    """
    Base class for every exception related to the AI model service
    (connection, request/streaming, shutdown, etc.).
    """
    def __init__(self, message: str = "AI model error") -> None:
        """
        Method for initializing the exception message

        Args:
            message (str): Description of the AI model error
        """
        super().__init__(message)


class AIModelConnectionError(AIModelError):
    """
    Custom exception for failures while opening a connection
    with the AI model provider.
    """
    def __init__(self, reason: str = None) -> None:
        """
        Method for initializing the exception message

        Args:
            reason (str, optional): Description of the underlying error
                that caused the connection to fail.
        """
        message = "Error while trying to connect to the AI model"
        if reason:
            message = f"{message}: {reason}"

        super().__init__(message)


class AIModelRequestError(AIModelError):
    """
    Custom exception for failures while requesting/streaming a
    response from the AI model.
    """
    def __init__(self, reason: str = None) -> None:
        """
        Method for initializing the exception message

        Args:
            reason (str, optional): Description of the underlying error
                that caused the request to fail.
        """
        message = "Error while requesting a response from the AI model"
        if reason:
            message = f"{message}: {reason}"

        super().__init__(message)


class FileProcessingError(AppError):
    """
    Base class for every exception related to loading or splitting
    files before they are vectorized.
    """
    def __init__(self, message: str = "File processing error") -> None:
        """
        Method for initializing the exception message

        Args:
            message (str): Description of the file processing error
        """
        super().__init__(message)


class PDFLoadError(FileProcessingError):
    """
    Custom exception for failures while parsing/loading the
    contents of a PDF file that does exist on disk.
    """
    def __init__(self, file_path: str, reason: str = None) -> None:
        """
        Method for initializing the exception message

        Args:
            file_path (str): Path of the PDF file that failed to load
            reason (str, optional): Description of the underlying error
                that caused the load to fail.
        """
        message = f"Error while trying to load PDF file: {file_path}"
        if reason:
            message = f"{message}: {reason}"

        super().__init__(message)


class FileSplitError(FileProcessingError):
    """
    Custom exception for failures while splitting documents
    into smaller chunks.
    """
    def __init__(self, reason: str = None) -> None:
        """
        Method for initializing the exception message

        Args:
            reason (str, optional): Description of the underlying error
                that caused the split to fail.
        """
        message = "Error while trying to split file contents"
        if reason:
            message = f"{message}: {reason}"

        super().__init__(message)


class VectorDatabaseError(AppError):
    """
    Custom exception for vector database-related errors
    """
    def __init__(self, message: str = "Vector database error") -> None:
        """
        Method for initializing the exception message

        Args:
            message (str): Description of the vector database error
        """
        super().__init__(message)


class EmbeddingError(VectorDatabaseError):
    """
    Custom exception for failures while building the embedding
    model used to vectorize documents.
    """
    def __init__(self, reason: str = None) -> None:
        """
        Method for initializing the exception message

        Args:
            reason (str, optional): Description of the underlying error
                that caused the embedding creation to fail.
        """
        message = "Error while trying to create the embedding model"
        if reason:
            message = f"{message}: {reason}"

        super().__init__(message)


class VectorDatabaseCreationError(VectorDatabaseError):
    """
    Custom exception for errors occurring during the attempt
    to create the database
    """
    def __init__(self, reason: str = None) -> None:
        """
        Method for initializing the exception message

        Args:
            reason (str, optional): Description of the underlying error
                that caused the database creation to fail. When provided,
                it is appended to the base message so the real cause is
                not hidden behind a generic error.
        """
        message = "Error while trying to create the database"
        if reason:
            message = f"{message}: {reason}"

        super().__init__(message)


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

    def _connect(self) -> genai.Client:
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


def get_path(file_name: str) -> str:
    """
    Build absolute path relative to the module directory.

    Args:
        file_name (str): Name or relative path of the file/directory

    Returns:
        str: Absolute path to the file/directory
    """
    return os.path.join(
        os.path.dirname(__file__),
        file_name
    )


class FileLoader(ABC):
    """
    Interface for manipulating files
    """
    @abstractmethod
    def loads_file(self) -> List[Document]:
        """
        Loads information from a file

        Returns:
            List[Document]: List of loaded document contents
        """
        pass


class PDFLoader(FileLoader):
    """
    Class to load PDF files
    """
    def __init__(self, file_path: str) -> None:
        """
        Method for initializing instance attributes

        Args:
            file_path (str): Absolute path of the PDF file to be loaded
        """
        self.file_path = file_path

        self.__post_init__()

    def __post_init__(self) -> None:
        """
        Checks if the PDF file exists at the given path

        Raises:
            PDFFileNotFoundError: If the PDF file is not found at the given path
        """
        if not os.path.exists(self.file_path):
            raise PDFFileNotFoundError(self.file_path)

    def loads_file(self) -> List[Document]:
        """
        Load the PDF document specified during initialization.

        Returns:
            List[Document]: List of loaded document pages

        Raises:
            PDFLoadError: if the PDF exists but fails to be parsed/loaded
        """
        logger.info(f"Loading PDF from: {self.file_path}")

        try:
            pdf_loader = PyPDFLoader(self.file_path)
            documents = pdf_loader.load()
        except Exception as e:
            raise PDFLoadError(self.file_path, str(e)) from e

        logger.info(f"Successfully loaded PDF: {len(documents)} pages")

        return documents


class FileSplitter:
    """
    Class to split file contents into smaller chunks
    """
    def __init__(self, file_loader: FileLoader) -> None:
        """
        Method for initializing instance attributes

        Args:
            file_loader (FileLoader): instance of a class that implements
                the FileLoader interface, used to load the file's contents
        """
        self.file_loader = file_loader

    def splits_file(self) -> List[Document]:
        """
        Split documents into smaller chunks for embedding and retrieval.

        Returns:
            List[Document]: List of document chunks with preserved metadata

        Raises:
            FileSplitError: if splitting the loaded documents fails
        """
        file_contents = self.file_loader.loads_file()

        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True
            )
            chunks = splitter.split_documents(file_contents)
        except Exception as e:
            raise FileSplitError(str(e)) from e

        logger.info(f"Created {len(chunks)} document chunks")

        return chunks


class EmbeddingProvider(ABC):
    """
    Interface for classes responsible for supplying an embedding model.

    Its single responsibility is deciding *which* embedding to use and
    how to initialize it — nothing about persisting or vectorizing
    documents lives here.
    """
    @abstractmethod
    def get_embedding(self):
        """
        Build and return an embedding model instance.

        Returns:
            Embeddings: An embedding model compatible with Chroma
        """
        pass


class FastEmbedEmbeddingProvider(EmbeddingProvider):
    """
    Supplies embeddings using FastEmbedEmbeddings.
    """
    def get_embedding(self):
        """
        Build and return a FastEmbedEmbeddings instance.

        Returns:
            FastEmbedEmbeddings: Initialized embedding model

        Raises:
            EmbeddingError: if the embedding model fails to initialize
        """
        logger.info("Initializing embeddings...")

        try:
            return FastEmbedEmbeddings()
        except Exception as e:
            raise EmbeddingError(str(e)) from e


class VectorDatabaseCreator:
    """
    Responsible solely for creating and persisting a Chroma vector
    database from document chunks.

    It does not decide which embedding to use (that's EmbeddingProvider's
    job) and it does not build the persist path (that's get_path's job) —
    it only receives those as collaborators and focuses on the single
    task of creating the database, translating any low-level failure
    into a VectorDatabaseCreationError.
    """
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        persist_directory: str
    ) -> None:
        """
        Method for initializing instance attributes

        Args:
            embedding_provider (EmbeddingProvider): supplies the
                embedding model to use when vectorizing documents
            persist_directory (str): path where the vector database
                will be persisted
        """
        self.embedding_provider = embedding_provider
        self.persist_directory = persist_directory

    def create(self, file_chunks: List[Document]) -> None:
        """
        Create a vector database (Chroma) from document chunks.

        Args:
            file_chunks (List[Document]): List of document chunks to vectorize

        Raises:
            VectorDatabaseCreationError: if the creation of the database fails
        """
        logger.info("Starting database creation pipeline")

        try:
            embedding = self.embedding_provider.get_embedding()

            logger.info(f"Creating vector database at: {self.persist_directory}")

            Chroma.from_documents(
                documents=file_chunks,
                embedding=embedding,
                persist_directory=self.persist_directory
            )

            logger.info("Vector database created successfully!")
            logger.info("Database pipeline completed successfully")

        except AppError:
            raise
        except Exception as e:
            raise VectorDatabaseCreationError(str(e)) from e


def main():
    pass


if __name__ == "__main__":
    main()