import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ._base import Distiller


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature,CL,idx,epc):
    T_max = 9
    T_min = 6
    if epc < 120:
        alp = epc / 240
    else:
        alp = 120 / 240
    T_min = (1 - alp) * T_min
    T_max = (1 - alp) * T_max

    # target = target.float()

    T = CL[idx, -1] * (T_max - T_min) + T_min
    a = T.shape
    T = torch.tensor(T).cuda(non_blocking=True).reshape(a[0], 1).float() 
    
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    
    pred_student = F.softmax(logits_student, dim=1)
    pred_teacher = F.softmax(logits_teacher/T, dim=1).float()
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher,  reduction="none")
    ).sum(1).mean()* (T.view(-1)**2)#T.view(-1)
    pred_teacher_part2 = F.softmax(
        logits_teacher/T - 1000.0 * gt_mask, dim=1
    ).float()
    log_pred_student_part2 = F.log_softmax(
        logits_student - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2,  reduction="none")
    ).sum(1).mean()* (T.view(-1)**2)
    
    
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(DKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        index = kwargs['index']
        epc = kwargs['epc']
        ten = kwargs['ten']
        logprobs = F.log_softmax(logits_student, dim=1)
        tmp_target = target.view(-1, 1)
        logpt = logprobs.gather(1, tmp_target)
        logpt = logpt.view(-1)
        pt_s = Variable(logpt.data.exp())
        ind_ = list(index)
        ind = []
        for i in ind_:
            ind.append(int(i))

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target,reduction='none')
        # print(loss_ce.shape)# 
        loss_student = loss_ce
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) *dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,ten,ind,epc
        )

        if (epc<=125 and epc%5==0) or epc==1:
            ten[ind,0] = (loss_student).cpu().detach().numpy().tolist()
            ten[ind,1] = 1-pt_s.cpu()
        # print(ten[ind[1],1],ten[ind[2],1])#+loss_dkd
        losses_dict = {
            "loss_ce": loss_ce.mean(),
            "loss_kd":  loss_dkd,
        }
        return logits_student, losses_dict
