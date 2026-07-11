import os


def get_path(file_name: str) -> str:
    """
    Build absolute path relative to the module directory.

    Args:
        file_name: Name or relative path of the file/directory

    Returns:
        Absolute path to the file/directory
    """
    return os.path.join(
        os.path.dirname(__file__),
        file_name
    )
