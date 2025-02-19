from .odis_agent import ODISAgent
from .odis_ns_agent import ODISNsAgent
from .model_agent import MODELAgent
from .model_nohigh_agent import MODELNHAgent
from .model_nolow_agent import MODELNLAgent
from .model_nopre_agent import MODELNPAgent
from .adapt_agent import ADAPTAgent
from .adapt_moco_agent import ADAPTMOCOAgent


REGISTRY = {}

REGISTRY["mt_odis"] = ODISAgent
REGISTRY["mt_odis_ns"] = ODISNsAgent
REGISTRY["mt_model"] = MODELAgent
REGISTRY["mt_model_nohigh"] = MODELNHAgent
REGISTRY["mt_model_nolow"] = MODELNLAgent
REGISTRY["mt_model_nopre"] = MODELNPAgent
REGISTRY["mt_adapt"] = ADAPTAgent
REGISTRY["mt_adapt_moco"] = ADAPTMOCOAgent
