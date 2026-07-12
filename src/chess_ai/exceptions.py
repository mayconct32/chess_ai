from logging_config import logger


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
    def __init__(self) -> None:
        """
        Method for initializing the exception message
        """
        super().__init__("API_KEY not found. Please set it in your .env file.")


class PDFFileNotFoundError(AppError):
    """
    Custom exception for file not found
    """
    def __init__(self, file_path: str) -> None:
        """
        Method for initializing the exception message

        Args:
            file_path (str): Path of the PDF file that was not found
        """
        super().__init__(f"PDF file not found: {file_path}")


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
    to create the database.
    """
    def __init__(self, reason: str = None) -> None:
        """
        Method for initializing the exception message.

        Args:
            reason (str, optional): Description of the underlying error
                that caused the database creation to fail.
        """
        message = "Error while trying to create the database"
        if reason:
            message = f"{message}: {reason}"

        super().__init__(message)


class VectorDatabaseQueryError(VectorDatabaseError):
    """
    Custom exception for failures while querying an existing
    vector database.
    """
    def __init__(self, reason: str = None) -> None:
        """
        Method for initializing the exception message.

        Args:
            reason (str, optional): Description of the underlying error
                that caused the query to fail.
        """
        message = "Error while trying to query the database"
        if reason:
            message = f"{message}: {reason}"

        super().__init__(message)
