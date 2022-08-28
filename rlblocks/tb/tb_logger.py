from collections import defaultdict
from typing import Dict, Union

from torch.utils.tensorboard import SummaryWriter


def merge_logs(logs_dict):
    result = defaultdict(dict)
    for log_tag, logs in logs_dict.items():
        for log_item_tag, val in logs.items():
            result[log_item_tag][log_tag] = val
    return result


class TensorboardLogger:

    def __init__(self, log_dir: str):
        self._writer = SummaryWriter(log_dir=log_dir)

    def log(self, log_data: Dict[str, Union[float, Dict[str, float]]], step: int = 0):
        for tag, value in log_data.items():
            if isinstance(value, dict):
                self._writer.add_scalars(tag, value, step)
            else:
                self._writer.add_scalar(tag, value, step)
