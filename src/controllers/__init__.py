REGISTRY = {}

from .basic_controller import BasicMAC
from .multi_task.mt_model_controller import MODELSMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["mt_model_mac"] = MODELSMAC
