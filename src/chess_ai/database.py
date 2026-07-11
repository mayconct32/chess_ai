from abc import ABC, abstractmethod
from typing import List

from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document

from .exceptions import AppError, EmbeddingError, VectorDatabaseCreationError
from .logging_config import logger


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
