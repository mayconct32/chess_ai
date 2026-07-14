import os
from pathlib import Path

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from chess_ai.ai_service import AIGeminiService, AIModelService
from chess_ai.config import API_KEY, DATABASE_DIR, GEMINI_VERSION
from chess_ai.database import (
    ChromaDatabaseCreator,
    ChromaDatabaseRepository,
    VectorDatabaseRepository,
)
from chess_ai.engine import ChessEngineService
from chess_ai.exceptions import AppError
from chess_ai.file_loader import FileSplitter, PDFLoader
from chess_ai.logging_config import configure_logging, logger
from chess_ai.prompt import prompt
from chess_ai.utils import get_path
from chess_ai.colors import ANSI_COLORS, RESET_COLORS


PDF_PATH = Path(get_path("../../chess.pdf")).resolve()

PERSIST_DIRECTORY = (
    Path(get_path("../../")).resolve() / DATABASE_DIR
).resolve()


def create_prompt(
    message: str,
    engine_service: ChessEngineService,
    repository: VectorDatabaseRepository,
) -> str:
    """
    Create a formatted prompt with context from the database and engine analysis.

    Args:
        message (str): The user's input message
        engine_service (ChessEngineService): The initialized chess engine service
        repository (VectorDatabaseRepository): The initialized vector database

    Returns:
        str: The formatted prompt ready for the AI model
    """
    PROMPT_TEMPLATE: ChatPromptTemplate = ChatPromptTemplate.from_template(prompt)

    analysis = engine_service.analyse(message)

    context = repository.get_relevant_chunks(message)

    formatted_prompt = PROMPT_TEMPLATE.invoke(
        {
            "context": context,
            "input": message,
            "engine_analysis": analysis,
        }
    )

    return formatted_prompt


def cli(
    ai_service: AIModelService,
    repository: VectorDatabaseRepository,
    engine_service: ChessEngineService,
) -> None:
    """
    Command Line Interface for interacting with the chess AI.

    Args:
        ai_service (AIModelService): The AI model service
        repository (VectorDatabaseRepository): The vector database repository
        engine_service (ChessEngineService): The chess engine service
    """
    ASCII_ART = """
 ██████╗██╗  ██╗███████╗███████╗███████╗     █████╗ ██╗
██╔════╝██║  ██║██╔════╝██╔════╝██╔════╝    ██╔══██╗██║
██║     ███████║█████╗  ███████╗███████╗    ███████║██║
██║     ██╔══██║██╔══╝  ╚════██║╚════██║    ██╔══██║██║
╚██████╗██║  ██║███████║███████║███████║    ██║  ██║██║
 ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝    ╚═╝  ╚═╝╚═╝
"""

    TEXT_INTERFACE = """
* AI assistant specialized in chess
* CTRL + C to quit
* Type a normal question to consult the database and AI model
* Type 'FEN: <fen_string>' to analyze a chess position with the engine
* Type 'exit' or 'quit' to leave
"""

    print(
        ANSI_COLORS['green'],
        ASCII_ART,
        RESET_COLORS,
        TEXT_INTERFACE,
    )

    while True:
        try:
            message = input(f"{ANSI_COLORS['cyan']}You:{RESET_COLORS} ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not message:
            continue

        if message.lower() in {"exit", "quit"}:
            break

        try:
            formatted_prompt = create_prompt(
                message,
                engine_service,
                repository,
            )

            print(
                f"{ANSI_COLORS['yellow']}Chess AI:{RESET_COLORS} ",
                end="",
                flush=True,
            )
            for chunk in ai_service.request(str(formatted_prompt)):
                print(chunk, end="", flush=True)
            print("\n")
        except AppError:
            print(
                f"{ANSI_COLORS['red']}An error occurred. "
                f"Check the logs for more details.{RESET_COLORS}"
            )
        except Exception:
            logger.exception("Unexpected error during chat iteration")
            print(
                f"{ANSI_COLORS['red']}An unexpected error occurred. "
                f"Check the logs for more details.{RESET_COLORS}"
            )


def gui() -> None:
    """
    Graphical User Interface for interacting with the chess AI.
    """
    pass


def main() -> None:
    """
    Entry point for the chess AI application.

    Handles initialization, interface selection, and top-level error handling.
    """
    configure_logging()

    try:
        if not PERSIST_DIRECTORY.exists() or not any(
            PERSIST_DIRECTORY.iterdir()
        ):
            file_chunks = FileSplitter(
                PDFLoader(str(PDF_PATH))
            ).splits_file()

            ChromaDatabaseCreator(
                FastEmbedEmbeddings(),
                str(PERSIST_DIRECTORY),
            ).create(file_chunks)

        repository = ChromaDatabaseRepository(
            FastEmbedEmbeddings(),
            str(PERSIST_DIRECTORY),
        )

        with (
            AIGeminiService(api_key=API_KEY, gemini_version=GEMINI_VERSION)
            as ai_service,
            ChessEngineService() as engine_service,
        ):
            response = input(
                f"{ANSI_COLORS['yellow']}Do you want to use our chess AI via "
                f"Command Line Interface or via graphical interface?(CLI/GUI): "
                f"{RESET_COLORS}"
            )

            os.system('cls' if os.name == 'nt' else 'clear')

            match response:
                case "CLI":
                    cli(
                        ai_service, 
                        repository, 
                        engine_service
                    )

                case "GUI":
                    gui()

                case _:
                    print(
                        f"{ANSI_COLORS['red']}This option is not currently "
                        f"available{RESET_COLORS}"
                    )

    except KeyboardInterrupt:
        pass
    except AppError:
        print(
            f"{ANSI_COLORS['red']}An error occurred during initialization. "
            f"Check the logs for more details.{RESET_COLORS}"
        )
    except Exception:
        logger.exception("Unexpected error during initialization")
        print(
            f"{ANSI_COLORS['red']}An unexpected error occurred. "
            f"Check the logs for more details.{RESET_COLORS}"
        )
    finally:
        print(f"\n{ANSI_COLORS['green']}See you next time!{RESET_COLORS}")


if __name__ == "__main__":
    main()
