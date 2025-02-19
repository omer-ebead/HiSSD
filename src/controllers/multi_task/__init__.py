from .mt_odis_controller import ODISMAC
from .mt_odis_ns_controller import ODISNSMAC
from .mt_model_controller import MODELSMAC
from .mt_model_nohigh_controller import MODELNHSMAC
from .mt_model_nolow_controller import MODELNLSMAC
from .mt_model_nopre_controller import MODELNPSMAC
from .mt_adapt_controller import ADAPTSMAC
from .mt_adapt_moco_controller import ADAPTMOCOSMAC


REGISTRY = {}

REGISTRY["mt_odis_mac"] = ODISMAC
REGISTRY["mt_odis_ns_mac"] = ODISNSMAC
REGISTRY["mt_model_mac"] = MODELSMAC
REGISTRY["mt_model_nohigh_mac"] = MODELNHSMAC
REGISTRY["mt_model_nolow_mac"] = MODELNLSMAC
REGISTRY["mt_model_nopre_mac"] = MODELNPSMAC
REGISTRY["mt_adapt_mac"] = ADAPTSMAC
REGISTRY["mt_adapt_moco_mac"] = ADAPTMOCOSMAC

