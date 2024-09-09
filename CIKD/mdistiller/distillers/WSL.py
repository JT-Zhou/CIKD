import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def wsl_loss(logits_student, logits_teacher, temperature, target):
    s_input = logits_student / temperature
    t_input = logits_teacher / temperature
    t_soft_label = F.softmax(t_input)
    softmax_loss = torch.sum(t_soft_label * F.log_softmax(s_input), 1, keepdim=True)
    fc_s_auto = logits_student.detach()
    fc_t_auto = logits_teacher.detach()
    log_softmax_s = F.log_softmax(fc_s_auto)
    log_softmax_t = F.log_softmax(fc_t_auto)
    one_hot_label = F.one_hot(target, num_classes=100).float()
    softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
    softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

    focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
    ratio_lower = torch.zeros(1).cuda()
    focal_weight = torch.max(focal_weight, ratio_lower)
    focal_weight = 1 - torch.exp(- focal_weight)
    softmax_loss = focal_weight * softmax_loss
    soft_loss = (temperature ** 2) * torch.mean(softmax_loss)
    return soft_loss


class WSL(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(WSL, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

        self.T = 2
        self.alpha = 2.5
        self.softmax = nn.Softmax(dim=1).cuda()
        self.logsoftmax = nn.LogSoftmax().cuda()
        self.hard_loss = nn.CrossEntropyLoss().cuda()

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        logits_teacher, _ = self.teacher(image)

        s_input = logits_student / self.temperature
        t_input = logits_teacher / self.temperature
        t_soft_label = self.softmax(t_input)
        softmax_loss = - torch.sum(t_soft_label * F.log_softmax(s_input), 1, keepdim=True)
        fc_s_auto = logits_student.detach()
        fc_t_auto = logits_teacher.detach()
        log_softmax_s = self.logsoftmax(fc_s_auto)
        log_softmax_t = self.logsoftmax(fc_t_auto)
        one_hot_label = F.one_hot(target, num_classes=100).float()
        softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
        softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        softmax_loss = focal_weight * softmax_loss
        loss_kd = (self.temperature ** 2) * torch.mean(softmax_loss)

        # losses
        loss_ce = F.cross_entropy(logits_student, target)  # self.ce_loss_weight *
        #loss_kd = wsl_loss(logits_student, logits_teacher, self.temperature, target)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": 2.5 * loss_kd,
        }
        return logits_student, losses_dict
