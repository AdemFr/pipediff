import pandas as pd


class DiffTracker:
    def __init__(self) -> None:
        self.frame_logs = {}
        self._frame_logs_counter = 0

    def log_frame(self, df: pd.DataFrame, name: str = None) -> None:
        name = self._get_unique_frame_name(name)
        self._frame_logs_counter += 1

        self.frame_logs[name] = "asd"

    def reset(self) -> None:
        self.frame_logs = {}
        self._frame_logs_counter = 0

    def _get_unique_frame_name(self, name: str = None) -> str:
        """If a name is given, checks if it exists, otherwise create a new one."""

        if name in self.frame_logs:
            raise KeyError(f"Key '{name}' already exists!")
        elif name is None:
            return f"df_{self._frame_logs_counter}"
        else:
            return name
