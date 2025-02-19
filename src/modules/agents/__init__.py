REGISTRY = {}

# normal agents
from .rnn_agent import RNNAgent
from .multi_task.model_agent import MODELAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["mt_model"] = MODELAgent
