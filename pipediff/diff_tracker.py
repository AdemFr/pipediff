import pandas as pd


class DiffTracker:
    def __init__(self) -> None:
        self.frame_logs = {}

    def log_frame(self, df: pd.DataFrame, name: str = None) -> None:
        if name is not None:
            self.frame_logs[name] = "asd"
        else:
            self.frame_logs[f"df_{len(self.frame_logs)}"] = "asd"

    def reset(self) -> None:
        self.frame_logs = {}
