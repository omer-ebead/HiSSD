from .odis_learner import ODISLearner
from .odis_ns_learner import ODISNsLearner
from .model_learner import MODELLearner
from .model_nohigh_learner import MODELNHLearner
from .model_nolow_learner import MODELNLLearner
from .model_nopre_learner import MODELNPLearner
from .adapt_learner import ADAPTLearner
from .adapt_moco_learner import ADAPTMOCOLearner

REGISTRY = {}

REGISTRY["odis_learner"] = ODISLearner
REGISTRY["odis_ns_learner"] = ODISNsLearner
REGISTRY["model_learner"] = MODELLearner
REGISTRY["model_nohigh_learner"] = MODELNHLearner
REGISTRY["model_nolow_learner"] = MODELNLLearner
REGISTRY["model_nopre_learner"] = MODELNPLearner
REGISTRY["adapt_learner"] = ADAPTLearner
REGISTRY["adapt_moco_learner"] = ADAPTMOCOLearner
