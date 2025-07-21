"""Function to append a number to a filename if chosen filename already exists to prevent overwriting"""

import os


def get_unique_filename(base: str, ext: str, root_dir: str):
    """
    Args:
        base: Desired filename
        ext: File extension
        root_dir: Root directory for file to be saved

    Returns:
        Unique filename with appended number if required
    """
    filename = os.path.join(root_dir, base + ext)
    index = 1
    while os.path.exists(filename):
        filename = os.path.join(root_dir, f"{base}_{index}{ext}")
        index += 1
    return filename