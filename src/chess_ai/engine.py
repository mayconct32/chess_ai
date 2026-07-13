import re
import threading
from dataclasses import dataclass
from typing import Optional

from chess import engine as chess_engine, Board

from chess_ai.config import ENGINE_ANALYSIS_TIME, STOCKFISH_PATH
from chess_ai.exceptions import (
    AppError,
    ChessEngineAnalysisError,
    ChessEngineInitializationError,
)
from chess_ai.logging_config import logger


@dataclass(frozen=True)
class PositionAnalysis:
    """
    Result of analysing a chess position with the engine.
    """
    board: Board
    info: chess_engine.InfoDict


def extract_fen(message: str) -> Optional[str]:
    """
    Extracts a FEN string from a chat message, if present.

    Args:
        message (str): user message potentially containing FEN notation

    Returns:
        Optional[str]: the extracted FEN string, or None if not found
    """
    FEN_REGEX = re.compile(
        r"(?<=FEN: )([prnbqkPRNBQK1-8]+(?:/[prnbqkPRNBQK1-8]+){7} [wb] "
        r"(?:-|[KQkq]{1,4}) (?:-|[a-h][36]) \d+ \d+)"
    )
    match = FEN_REGEX.search(message)
    return match.group() if match else None


def validate_fen(fen: Optional[str]) -> Optional[Board]:
    """
    Validates a FEN string and builds the corresponding board.

    Args:
        fen (Optional[str]): FEN string to validate; may be None when
            no FEN was found in the original message

    Returns:
        Optional[Board]: the resulting board if the FEN is valid,
            None if it is missing or invalid
    """
    if not fen:
        return None

    try:
        board = Board(fen)
    except ValueError:
        return None

    return board if board.is_valid() else None


class ChessEngineService:
    """
    Class to analyze chess positions using a UCI-compatible chess engine
    """
    def __init__(self, engine_path: str = STOCKFISH_PATH) -> None:
        """
        Method for initializing instance attributes

        Args:
            engine_path (str): path or command used to launch the
                UCI-compatible chess engine
        """
        self.engine_path = engine_path
        self.engine = None
        self._lock = threading.Lock()

    def __enter__(self):
        """
        Enables use of the class as a context manager (with-statement).
        The connection itself is opened lazily on first use, not here.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Ensures the engine process is closed when leaving a with-block,
        even if an exception occurred inside it.
        """
        self.close()

    def _connect(self) -> chess_engine.SimpleEngine:
        """
        Initialize and return a connection to the chess engine.

        Returns:
            SimpleEngine: Initialized engine instance

        Raises:
            ChessEngineInitializationError: if the engine fails to initialize
        """
        logger.info(f"Starting {self.engine_path} engine")

        try:
            engine = chess_engine.SimpleEngine.popen_uci(self.engine_path)
        except Exception as e:
            raise ChessEngineInitializationError(str(e)) from e

        logger.info(f"Successfully initialized {self.engine_path} engine")
        return engine

    def analyse(self, message: str) -> Optional[PositionAnalysis]:
        """
        Analyze the chess position described by a FEN found in a message.

        Opens a connection to the engine immediately if one doesn't
        exist yet (protected by a lock against concurrent calls), then
        reuses it for every subsequent analysis.

        Args:
            message (str): user message potentially containing FEN notation

        Returns:
            Optional[PositionAnalysis]: the board and the engine's analysis
            result when a valid FEN is available, otherwise None

        Raises:
            ChessEngineInitializationError: if opening the engine fails
            ChessEngineAnalysisError: if the engine analysis fails
        """
        fen = extract_fen(message)
        if not fen:
            logger.info("No FEN string found in message; skipping analysis")
            return None

        board = validate_fen(fen)
        if not board:
            logger.info("Provided FEN string is invalid; skipping analysis")
            return None

        with self._lock:
            if not self.engine:
                self.engine = self._connect()

        try:
            limit = chess_engine.Limit(time=ENGINE_ANALYSIS_TIME)
            info = self.engine.analyse(board, limit)

        except AppError:
            raise
        except Exception as e:
            raise ChessEngineAnalysisError(str(e)) from e

        logger.info("Successfully analyzed position")
        
        return PositionAnalysis(board=board, info=info)

    def close(self) -> None:
        """
        Method for closing the connection with the chess engine.

        Protected by a lock to avoid closing an engine that another
        thread is in the middle of creating.

        Raises:
            ChessEngineInitializationError: if closing the engine fails
        """
        with self._lock:
            if self.engine:
                logger.info("Closing connection with the chess engine")
                try:
                    self.engine.quit()
                except Exception as e:
                    raise ChessEngineInitializationError(str(e)) from e
                finally:
                    self.engine = None