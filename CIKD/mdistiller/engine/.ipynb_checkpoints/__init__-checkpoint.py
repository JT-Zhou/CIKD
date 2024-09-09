from .trainer import BaseTrainer, CRDTrainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
}

from .trainer_ts import BaseTrainer,CRDTrainer

trainer_ts_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
}