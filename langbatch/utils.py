import os
import logging
from pathlib import Path

def get_data_path():
    # Default data path, can be overridden by environment variable
    data_path = Path(__file__).parent.parent / "data"

    def check_path_writable(path):
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        test_file = path / "test.txt"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except (OSError, PermissionError) as e:
            raise e

    # Check custom path if provided
    langbatch_data_path = os.environ.get("LANGBATCH_DATA_PATH")
    if langbatch_data_path:
        custom_path = Path(langbatch_data_path)
        try:
            check_path_writable(custom_path)
            return custom_path
        except Exception as e:
            logging.warning(f"Custom data path is not writable: {langbatch_data_path}. Error: {e}. Falling back to default path.")

    # Check default path
    try:
        check_path_writable(data_path)
        return data_path
    except Exception as e:
        raise PermissionError(f"Unable to write to default data path: {data_path}. Error: {e}")