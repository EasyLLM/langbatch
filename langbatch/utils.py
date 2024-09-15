import os
import logging
from pathlib import Path

def get_data_path():
    # Default data path, can be overridden by environment variable
    DEFAULT_DATA_PATH = Path(__file__).parent / "data"
    DATA_PATH = DEFAULT_DATA_PATH

    langbatch_data_path = os.environ.get("LANGBATCH_DATA_PATH")
    if langbatch_data_path:
        try:
            data_path = Path(langbatch_data_path)
            test_file = data_path / "test.txt"
            # test if the directory is writable
            with open(test_file, 'w') as f:
                f.write("test")

            test_file.unlink(missing_ok=True)
            DATA_PATH = langbatch_data_path
        except:
            logging.warning(f"Invalid data path: {langbatch_data_path}, using default data path: {DEFAULT_DATA_PATH}")

    return DATA_PATH