import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def temp(logit):
    bs,C = logit.shape[0],logit.shape[1]
    # print(logit.norm(dim=1).mean(),logit.std(dim=1).mean())
    stdv = logit.std(dim=-1, keepdims=True)
    return stdv+1e-7#(logit.std(dim=1)+1e-7).reshape(bs,1)

def kd_loss(logits_student, logits_teacher, temperature):
    temp_t = temp(logits_teacher)
    temp_s = temp(logits_student)
    # print(temp_t,temp_s)
    log_pred_student = F.log_softmax((logits_student-logits_student.mean(dim=-1, keepdims=True)) / (temp_s*temperature), dim=1)
    pred_teacher = F.softmax((logits_teacher-logits_teacher.mean(dim=-1, keepdims=True)) / (temp_t*temperature), dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)#.mean()
    # print(loss_kd.shape,(temp_s*temp_t).shape)
    # print(log_pred_student.topk(5))
    loss_kd *= temperature**2#(temp_s.view(-1)+temperature)*(temp_t.view(-1)+temperature)#(temp_s.view(-1)*temp_t.view(-1))
    # loss_kd *= (1+temp_s).view(-1)
    # print((temp_s*temperature).mean(),(temp_t*temperature).mean(),temp_s.mean(),temp_t.mean())
    return loss_kd.mean()


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = 0.1 * F.cross_entropy(logits_student, target)
        loss_kd = 0.9 * kd_loss(
            logits_student, logits_teacher, 4
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
    