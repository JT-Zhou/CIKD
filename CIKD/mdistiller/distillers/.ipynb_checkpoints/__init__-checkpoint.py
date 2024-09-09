from ._base import Vanilla
from .AT import AT
from .KD_clip import KD_clip
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .VID import VID
from .IAKD import IAKD
from .DKD_M import DKD_M
from .DKD_v import DKD
from .KD_M import KD_M
from .KD import KD
from .WSL import WSL
from .ICKD import ICKD
from .CTKD import CTKD
from .DKD_clip import DKD_clip

distiller_dict = {
    "NONE": Vanilla,
    "KD_clip": KD_clip,
    "KD": KD,
    "DKD_clip": DKD_clip,
    "DKD": DKD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "VID": VID,
    "IAKD": IAKD,
    "DKD_M": DKD_M,
    "KD_M":KD_M,
    "WSL":WSL,
    "ICKD":ICKD,
    'CTKD':CTKD
}
