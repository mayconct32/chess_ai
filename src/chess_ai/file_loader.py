import os
from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from exceptions import PDFFileNotFoundError, PDFLoadError, FileSplitError
from logging_config import logger


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
