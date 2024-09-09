import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ._base1 import Distiller
import jenkspy
from numpy import *
from ._common import ConvReg, get_feat_shapes
from torch.autograd import Variable


class CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''

    def __init__(self, label_smooth=None, class_num=100):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12

        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)  # softmax + log
            tmp = target.view(-1, 1)
            target = F.one_hot(target, self.class_num)  # 转换成one-hot
            # tmp = target.view(-1,1)
            logpt = logprobs.gather(1, tmp)
            logpt = logpt.view(-1)
            pt = Variable(logpt.data.exp())

            # label smoothing
            # 实现 1
            target = (1.0 - self.label_smooth) * target + self.label_smooth / self.class_num
            # 实现 2
            # implement 2
            # target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1),max=1.0 - self.label_smooth)

            loss = -1 * (1 - pt) * torch.sum(target * logprobs, 1)  #

        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))

        return loss.mean()


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
        alpha = 100/240
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

    target = target.float()
    CL_S = CL[idx,-1]
    
    # for i in range(len(idx)):
        # CL_S.append(CL[idx[i],-1])
    T = CL[idx,-1]*(T_max-T_min)+T_min
    a = T.shape
    T = torch.tensor(T).cuda(non_blocking=True).reshape(a[0],1)
    # print(T.shape)
    logits_teacher = torch.div(logits_teacher,T).float()
    # logits_student[i] = torch.div(logits_student[i],T)
    # tmp.append(T)#**
        
    
    CL_S = np.array(CL_S)
    CL_S = torch.tensor(CL_S).cuda(non_blocking=True)
    log_pred_student = F.log_softmax(logits_student, dim=1)
    pred_teacher = F.softmax(logits_teacher, dim=1)
    loss_kd = F.kl_div(log_pred_student,pred_teacher, reduction="none").sum(1)
    tmp = torch.tensor(tmp).cuda(non_blocking=True)
    loss_kd = torch.mul(loss_kd,T)#*alpha#*(1-pt)
    loss_kd = torch.mul(loss_kd,CL_S+0.5)
    # loss_kd = torch.mul(loss_kd,2*tmp+1)#*alpha#*(1-pt)
    loss_kd = loss_kd.mean()*4
    # loss = -1*CL_S*torch.sum(target * logprobs, 1)
    return loss_kd



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
            logits_student, logits_teacher, ten, ind, epc, target, self.T_MAX, self.T_MIN, self.Reduce
        )

        # if epc in [1, 25, 50, 75, 100, 125]:
        #     # l_kd_ = list(l_kd_each)
        #     for i in range(len(loss_student)):
        #         ten[ind[i], 0] = loss_student[
        #             i]  # self.ce_loss_weight*loss_student[i]+self.kd_loss_weight*float(l_kd_[i])
        #         ten[ind[i], 1] = 1 - pt_s[i]
        losses_dict = {
            "loss_kd": b * loss_kd,
            "loss_ce_": loss_ce
        }  # "loss_ce": a*loss_ce,
        return logits_student, losses_dict

    def forward_test(self,**kwargs):
        # print(kwargs)
        if kwargs['mode'] =='tmp':
            with torch.no_grad():
                logits_teacher, feature_teacher = self.teacher(kwargs['image'])
                logits_student, feature_student = self.student(kwargs['image'])
            index = kwargs['index']
            epc = kwargs['epc']
            ten = kwargs['ten']
            target = kwargs['target']
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
            # loss_student = ce_loss(logits_student, target)
            loss_ce = 1 * F.cross_entropy(logits_student, target,reduction="none")
            loss_student = loss_ce.cpu().detach().numpy().tolist()
            a = 1
            b = min(kwargs["epc"] / 20, 1)
            loss_kd = kd_loss(
                logits_student, logits_teacher,ten, ind, epc,target,self.T_MAX,self.T_MIN,self.Reduce
            )

            if (epc%25==0 and epc<=125) or epc==1:
                #l_kd_ = list(l_kd_each)
                # for i in range(len(loss_student)):
                ten[ind,0] = loss_student #self.ce_loss_weight*loss_student[i]+self.kd_loss_weight*float(l_kd_[i])epc %25==0 and 
                ten[ind,1] = 1-pt_s.cpu()
        else:
            return self.student(kwargs['image'])[0]