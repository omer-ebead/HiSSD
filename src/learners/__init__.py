REGISTRY = {}

# normal learner
from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .multi_task.model_learner import MODELLearner

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["model_learner"] = MODELLearner
REGISTRY["qtran_learner"] = QTranLearner
