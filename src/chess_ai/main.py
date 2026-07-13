from pathlib import Path

from langchain_community.embeddings import FastEmbedEmbeddings

from chess_ai.ai_service import AIGeminiService
from chess_ai.config import API_KEY, DATABASE_DIR, GEMINI_VERSION
from chess_ai.database import ChromaDatabaseCreator, ChromaDatabaseRepository
from chess_ai.engine import ChessEngineService
from chess_ai.exceptions import (
    AppError,
    ChessEngineAnalysisError,
    ChessEngineInitializationError,
)
from chess_ai.file_loader import FileSplitter, PDFLoader
from chess_ai.logging_config import configure_logging
from chess_ai.utils import get_path


ANSI_COLORS = {
    "green": "\033[92m",
    "cyan": "\033[96m",
    "yellow": "\033[93m",
    "red": "\033[91m",
}

RESET_COLORS = "\033[0m"

ASCII_ART = """
 ██████╗██╗  ██╗███████╗███████╗███████╗     █████╗ ██╗
██╔════╝██║  ██║██╔════╝██╔════╝██╔════╝    ██╔══██╗██║
██║     ███████║█████╗  ███████╗███████╗    ███████║██║
██║     ██╔══██║██╔══╝  ╚════██║╚════██║    ██╔══██║██║
╚██████╗██║  ██║███████║███████║███████║    ██║  ██║██║
 ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝    ╚═╝  ╚═╝╚═╝
"""

PDF_PATH = Path(get_path("../../chess.pdf")).resolve()
PERSIST_DIRECTORY = (
    Path(get_path("../../")).resolve() / DATABASE_DIR
).resolve()


def run_interactive_session(
    ai_service: AIGeminiService,
    repository: ChromaDatabaseRepository,
    engine_service: ChessEngineService,
) -> None:
    print("* Type a normal question to consult the database and AI model")
    print(
        "* Type 'FEN: <fen_string>' to analyze a chess position "
        "with the engine"
    )
    print("* Type 'exit' or 'quit' to leave\n")

    while True:
        message = input(f"{ANSI_COLORS['cyan']}You:{RESET_COLORS} ").strip()
        if not message:
            continue

        if message.lower() in {"exit", "quit"}:
            break

        try:
            if message.upper().startswith("FEN:"):
                fen_value = message[4:].strip()
                if not fen_value:
                    continue

                try:
                    analysis = engine_service.analyse(message)
                except (
                    ChessEngineInitializationError,
                    ChessEngineAnalysisError,
                ):
                    continue

                if analysis is None:
                    continue

                print(
                    f"{ANSI_COLORS['yellow']}Engine analysis result:"
                    f"{RESET_COLORS}\n"
                )
                print(analysis.board)
                print("\nInfo:\n")
                print(analysis.info)
                print("\n")
                continue

            context = repository.get_relevant_chunks(message)
            prompt = f"Context:\n{context}\n\nQuestion:\n{message}"

            print(
                f"{ANSI_COLORS['yellow']}Chess AI:{RESET_COLORS} ",
                end="",
                flush=True,
            )
            for chunk in ai_service.request(prompt):
                print(chunk, end="", flush=True)
            print("\n")
        except AppError:
            print(
                f"{ANSI_COLORS['red']}Something went wrong. "
                f"Check the logs above.{RESET_COLORS}"
            )
            continue


def main() -> None:
    configure_logging()

    print(f"{ANSI_COLORS['green']}{ASCII_ART}{RESET_COLORS}")
    print("* AI assistant specialized in chess")
    print("* CTRL + C to quit\n")

    try:
        if not PERSIST_DIRECTORY.exists() or not any(
            PERSIST_DIRECTORY.iterdir()
        ):
            file_chunks = FileSplitter(PDFLoader(str(PDF_PATH))).splits_file()
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
            run_interactive_session(ai_service, repository, engine_service)
    except AppError:
        print(
            f"{ANSI_COLORS['red']}Something went wrong. "
            f"Check the logs above.{RESET_COLORS}"
        )
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n{ANSI_COLORS['green']}See you next time!{RESET_COLORS}")


if __name__ == "__main__":
    main()