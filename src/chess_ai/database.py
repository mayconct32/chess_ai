import threading
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma

from config import DATABASE_DIR, SIMILARITY_SCORE_THRESHOLD, SIMILARITY_SEARCH_K
from exceptions import AppError, VectorDatabaseCreationError, VectorDatabaseQueryError
from logging_config import logger


class VectorDatabaseCreator(ABC):
    """
    Interface for creating and persisting vector databases
    """
    @abstractmethod
    def create(self, file_chunks: List[Document]) -> None:
        """
        Creates and persists a vector database from document chunks

        Args:
            file_chunks (List[Document]): document chunks to be vectorized

        Raises:
            VectorDatabaseCreationError: if the database creation fails
        """
        pass


class VectorDatabaseRepository(ABC):
    """
    Interface for querying an existing vector database
    """
    @abstractmethod
    def get_relevant_chunks(self, message: str) -> str:
        """
        Retrieves the most relevant document chunks for a given message

        Args:
            message (str): user query used to search the database

        Returns:
            str: concatenated content of the relevant chunks, joined by
                a separator, or an empty string if none are found

        Raises:
            VectorDatabaseQueryError: if the query fails
        """
        pass


class ChromaDatabaseCreator(VectorDatabaseCreator):
    """
    Class to create and persist a Chroma vector database
    """
    def __init__(self, embedding_provider: Any, persist_directory: str = DATABASE_DIR) -> None:
        """
        Method for initializing instance attributes

        Args:
            embedding_provider (Any): provider exposing a get_embedding()
                method, used to build the embedding function
            persist_directory (str): directory where the database will
                be persisted
        """
        self.embedding_provider = embedding_provider
        self.persist_directory = persist_directory

    def create(self, file_chunks: List[Document]) -> None:
        """
        Creates and persists a Chroma vector database from document chunks.

        Args:
            file_chunks (List[Document]): document chunks to be vectorized

        Raises:
            VectorDatabaseCreationError: if no chunks are provided or if
                the database creation fails
        """
        if not file_chunks:
            raise VectorDatabaseCreationError("No document chunks were provided")

        logger.info("Starting database creation pipeline")

        try:
            embedding = self.embedding_provider
            logger.info(f"Creating vector database at: {self.persist_directory}")

            Chroma.from_documents(
                documents=file_chunks,
                embedding=embedding,
                persist_directory=self.persist_directory,
            )

            logger.info("Vector database created successfully!")
            logger.info("Database pipeline completed successfully")
        except AppError:
            raise
        except Exception as e:
            raise VectorDatabaseCreationError(str(e)) from e


class ChromaDatabaseRepository(VectorDatabaseRepository):
    """
    Class to query an existing Chroma vector database
    """
    def __init__(
        self,
        embedding_provider: Any,
        persist_directory: str = DATABASE_DIR,
        similarity_search_k: int = SIMILARITY_SEARCH_K,
        similarity_score_threshold: float = SIMILARITY_SCORE_THRESHOLD,
    ) -> None:
        """
        Method for initializing instance attributes

        Args:
            embedding_provider (Any): provider exposing a get_embedding()
                method, used to build the embedding function needed to
                open the persisted database
            persist_directory (str): directory where the database is
                persisted
            similarity_search_k (int): number of chunks to retrieve per query
            similarity_score_threshold (float): minimum relevance score a
                chunk must have to be included in the result
        """
        self.embedding_provider = embedding_provider
        self.persist_directory = persist_directory
        self.similarity_search_k = similarity_search_k
        self.similarity_score_threshold = similarity_score_threshold
        self._db: Optional[Chroma] = None
        self._lock = threading.Lock()

    def _connect(self) -> Chroma:
        """
        Opens a connection to the persisted Chroma database.

        Returns:
            Chroma: initialized Chroma database instance

        Raises:
            VectorDatabaseQueryError: if the database cannot be opened
        """
        logger.info(f"Opening vector database at: {self.persist_directory}")

        try:
            embedding = self.embedding_provider

            from langchain_chroma.vectorstores import Chroma

            return Chroma(
                embedding_function=embedding,
                persist_directory=self.persist_directory,
            )
        except Exception as e:
            raise VectorDatabaseQueryError(str(e)) from e

    def get_relevant_chunks(self, message: str) -> str:
        """
        Retrieves the most relevant document chunks for a given message.

        Opens a connection to the persisted database lazily, on first use.

        Args:
            message (str): user query used to search the database

        Returns:
            str: concatenated content of the relevant chunks, joined by
                a separator, or an empty string if none are found

        Raises:
            VectorDatabaseQueryError: if the message is empty or the
                query against the database fails
        """
        if not message or not message.strip():
            raise VectorDatabaseQueryError("The search message cannot be empty")

        with self._lock:
            if self._db is None:
                self._db = self._connect()
            db = self._db

        try:
            results = db.similarity_search_with_relevance_scores(
                message,
                k=self.similarity_search_k,
            )
        except AppError:
            raise
        except Exception as e:
            raise VectorDatabaseQueryError(str(e)) from e

        if not results:
            logger.info("No relevant chunks found for the provided query")
            return ""

        result_content = [
            doc.page_content
            for doc, score in results
            if score >= self.similarity_score_threshold
        ]

        information = "\n\n----\n\n".join(result_content)

        logger.info(f"Retrieved {len(result_content)} relevant chunks from database")
        
        return information