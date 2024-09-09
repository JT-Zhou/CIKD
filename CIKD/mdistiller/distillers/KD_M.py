import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller_M


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class KD_M(Distiller_M):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher1, teacher2,cfg):
        super(KD_M, self).__init__(student, teacher1,teacher2)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher1, _ = self.teacher1(image)
            logits_teacher2, _ = self.teacher2(image)
            
        # losses
        loss_ce = 0.1 * F.cross_entropy(logits_student, target)
        loss_kd = 0.9 * kd_loss(
            logits_student, logits_teacher1, self.temperature
        )
        loss_kd2 = 1.3 * kd_loss(
            logits_student, logits_teacher2, self.temperature
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd#+loss_kd2,
        }
        return logits_student, losses_dict