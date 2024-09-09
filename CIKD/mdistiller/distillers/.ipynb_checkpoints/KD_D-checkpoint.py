import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ._base import Distiller
import jenkspy
from numpy import *
from ._common import ConvReg, get_feat_shapes
from torch.autograd import Variable





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


def label_smoothing(inputs, epsilon=0.1):
    K = inputs.shape[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


def ce_loss(hid, target):
    target = target.reshape(hid.shape[0], 1)
    log_pro = -1.0 * F.log_softmax(hid, dim=1)
    one_hot = torch.zeros(hid.shape[0], hid.shape[1]).cuda()
    one_hot = one_hot.scatter_(1, target, 1)
    loss_our = torch.mul(log_pro, one_hot).sum(dim=1)
    L = loss_our.cpu().detach().numpy().tolist()
    return L


def kd_loss(logits_student, logits_teacher, CL, idx, epc,target,T_MAX,T_MIN,Reduce):
    T_max = T_MAX
    T_min = T_MIN
    alpha = 0
    #sm = [0, 0.04, 0.06, 0.08, 0.1]
    # if epc > 25 and epc<=50 :
    #     alpha = 25/240
    # elif epc > 50 and epc<=75:
    #     alpha = 50/240
    # elif epc > 75 and epc<=100:
    #     alpha = 75/240
    if epc < 120:
        alpha = epc/240
    else:
        alpha = 120/240
    T_min = Reduce*(1-alpha)*T_min
    T_max = Reduce*(1-alpha)*T_max
    tmp = []
    if T_min<2:
        T_min=2
    if T_max<2:
        T_max=2
    # logprobs = F.log_softmax(logits_student, dim=1)
    # tmp_target = target.view(-1, 1)
    # logpt = logprobs.gather(1, tmp_target)
    # logpt = logpt.view(-1)
    # pt_s = Variable(logpt.data.exp())
    # target = F.one_hot(target, 100)  # 转换成one-hot

    # target = target.float()
    CL_S = CL[idx,-1]
    CL_S = np.array(CL_S)
    CL_S = torch.tensor(CL_S).cuda(non_blocking=True).float()
    # print(CL_S.shape)
    # for i in range(len(idx)):
        # CL_S.append(CL[idx[i],-1])
    T = CL[idx,-1]*(T_max-T_min)+T_min
    a = T.shape
    T = torch.tensor(T).cuda(non_blocking=True).reshape(a[0],1)
    CL_S = CL_S.reshape(a[0],1)
    # print(T.shape)

    
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    
    pred_student = F.softmax(logits_student, dim=1)
    pred_teacher = F.softmax(logits_teacher/T, dim=1).float()
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher,  reduction="none")
        *(CL_S+1)
    ).sum(1).mean()* (T.view(-1)**2)
    pred_teacher_part2 = F.softmax(
        logits_teacher/T - 1000.0 * gt_mask, dim=1
    ).float()
    log_pred_student_part2 = F.log_softmax(
        logits_student - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2,  reduction="none")
        *(CL_S+1)
    ).sum(1).mean()* (T.view(-1)**2)
    # logits_student[i] = torch.div(logits_student[i],T)(T/(CL_S+1))
    # tmp.append(T)#**
        
    # logits_student = torch.div(logits_student,T).float()
    # logits_teacher = torch.div(logits_teacher,T).float()  
    # CL_S = np.array(CL_S)
    # CL_S = torch.tensor(CL_S).cuda(non_blocking=True)
    # log_pred_student = F.log_softmax(logits_student, dim=1)
    # pred_teacher = F.softmax(logits_teacher, dim=1)
    # loss_kd = F.kl_div(log_pred_student,pred_teacher, reduction="none").sum(1)
    # tmp = torch.tensor(tmp).cuda(non_blocking=True)
    # loss_kd = torch.mul(loss_kd,T)#*alpha#*(1-pt)
    # loss_kd = torch.mul(loss_kd,CL_S+0.5)
    # loss_kd = torch.mul(loss_kd,2*tmp+1)#*alpha#*(1-pt)
    # loss_kd = loss_kd.mean()*4
    # loss = -1*CL_S*torch.sum(target * logprobs, 1)
    loss_kd = 1 * tckd_loss + 8 * nckd_loss
    # loss_kd = F.kl_div(log_pred_student_part2,pred_teacher_part2, reduction="none").sum(1)
    return loss_kd.to(torch.float32)



class KD_(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD_, self).__init__(student, teacher)
        self.temperature = cfg.KD_.TEMPERATURE
        self.ce_loss_weight = cfg.KD_.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD_.LOSS.KD_WEIGHT
        self.T_MAX = cfg.KD_.T_MAX
        self.T_MIN = cfg.KD_.T_MIN
        self.loss_ = None
        self.Reduce = cfg.KD_.Reduce
    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)
        index = kwargs['index']
        epc = kwargs['epc']
        ten = kwargs['ten']
        # losses

        logprobs = F.log_softmax(logits_student, dim=1)
        tmp_target = target.view(-1, 1)
        logpt = logprobs.gather(1, tmp_target)
        logpt = logpt.view(-1)
        pt_s = Variable(logpt.data.exp())

        ind_ = list(index)
        ind = []
        for i in ind_:
            ind.append(int(i))
        loss_student = ce_loss(logits_student, target)
        loss_ce = 1 * F.cross_entropy(logits_student, target)
        a = 1
        b = min(kwargs["epc"] / 20, 1)
        loss_kd = kd_loss(
            logits_student, logits_teacher,ten, ind, epc,target,self.T_MAX,self.T_MIN,self.Reduce
        )

        # if epc in [1,25,50,75,100,125]:
        # if (epc<=125 and epc%25==0) or epc==1:
        #     #l_kd_ = list(l_kd_each)
        #     for i in range(len(loss_student)):
        #         ten[ind[i],0] = loss_student[i] #self.ce_loss_weight*loss_student[i]+self.kd_loss_weight*float(l_kd_[i])
        #         ten[ind[i],1] = 1-pt_s[i]
        if (epc<=125 and epc%5==0) or epc==1:
            ten[ind,0] = loss_student#.cpu().detach().numpy().tolist() #+0.9*loss_kd).cpu().detach().numpy().tolist()
            ten[ind,1] = 1-pt_s.cpu()
        
        losses_dict = {
            "loss_kd": b*loss_kd,
            "loss_ce_":loss_ce
        }#"loss_ce": a*loss_ce,
        return logits_student, losses_dict
