import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
import pandas as pd
from torch.autograd import Variable
from ._base import Distiller
import numpy as np

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


            #tmp = target.view(-1,1)
            logpt = logprobs.gather(1,tmp)
            logpt = logpt.view(-1)
            pt = Variable(logpt.data.exp())

            # label smoothing
            # 实现 1
            # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num
            # 实现 2
            # implement 2
            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1),
                                 max=1.0 - self.label_smooth)
            loss = -1 *(1-pt)* torch.sum(target * logprobs, 1)#

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



def kd_loss(logits_student, logits_teacher,f_s,f_t,idx, epc,target,CL,T_MAX,T_MIN,Reduce,pt_s):
    T_max = T_MAX
    T_min = T_MIN
    alpha = 0
    #sm = [0, 0.04, 0.06, 0.08, 0.1]
    if epc > 25 and epc<=50 :
        alpha = 25/240
    elif epc > 50 and epc<=75:
        alpha = 50/240
    elif epc > 75 and epc<=100:
        alpha = 75/240
    elif epc > 100 and epc<=240:
        alpha = 100/240
    T_min = Reduce*(1-alpha)*T_min
    T_max = Reduce*(1-alpha)*T_max
    tmp = []
    if T_min<2:
        T_min=2
    if T_max<2:
        T_max=2
    logprobs = F.log_softmax(logits_student, dim=1)
    # tmp_target = target.view(-1, 1)
    # logpt = logprobs.gather(1, tmp_target)
    # logpt = logpt.view(-1)
    # pt_s = Variable(logpt.data.exp())
    target = F.one_hot(target, 100)  # 转换成one-hot
    pt = 1-pt_s
    target = target.float()
    CL_S = []
    for i in range(len(idx)):
        CL_S.append(CL[idx[i],-1])
        T = CL[idx[i],-1]*(T_max-T_min)+T_min
        label_smooth= CL[idx[i],-1]/10#sm[int(CL[idx[i],-1])]
        logits_teacher[i] = torch.div(logits_teacher[i],T)
        if label_smooth==0:
            target[i] = target[i]
        else:
            target[i] = torch.mul(target[i],(1-label_smooth))
            target[i] = torch.add(target[i],label_smooth/100)
        tmp.append(T)#**
    CL_S = np.array(CL_S)
    CL_S = torch.tensor(CL_S).cuda(non_blocking=True)
    log_pred_student = F.log_softmax(logits_student, dim=1)
    pred_teacher = F.softmax(logits_teacher, dim=1)

    loss_all = 0.0
    for fs, ft in zip(f_s, f_t):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            tmploss = F.mse_loss(tmpfs,tmpft,reduction='none')
            for i in range(len(idx)):
                tmploss[i] = torch.mul(tmploss[i],pt[i])
                tmploss[i] = torch.mul(tmploss[i],CL[idx[i],-1]+0.5)
            tmploss = tmploss.mean()
            cnt /= 2.0
            loss += tmploss*cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss


    loss_kd = F.kl_div(log_pred_student,pred_teacher, reduction="none").sum(1)
    tmp = torch.tensor(tmp).cuda(non_blocking=True)
    loss_kd = torch.mul(loss_kd,tmp).mean()
    loss = -1 * CL_S * torch.sum(target * logprobs, 1)
    loss = loss.mean()
    return loss_kd,loss,loss_all

def hcl_loss(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all


class IAKD(Distiller):
    def __init__(self, student, teacher, cfg,dim):
        super(IAKD, self).__init__(student, teacher)
        self.shapes = dim[0]#cfg.REVIEWKD.SHAPES
        self.out_shapes = dim[1]#cfg.REVIEWKD.OUT_SHAPES
        in_channels = dim[2]#cfg.REVIEWKD.IN_CHANNELS
        out_channels = dim[3]#cfg.REVIEWKD.OUT_CHANNELS
        self.T_MAX = cfg.KD_.T_MAX
        self.T_MIN = cfg.KD_.T_MIN
        self.Reduce = cfg.KD_.Reduce
        self.ce_loss_weight = cfg.REVIEWKD.CE_WEIGHT
        self.reviewkd_loss_weight = cfg.REVIEWKD.REVIEWKD_WEIGHT
        self.warmup_epochs = cfg.REVIEWKD.WARMUP_EPOCHS
        self.stu_preact = cfg.REVIEWKD.STU_PREACT
        self.max_mid_channel = cfg.REVIEWKD.MAX_MID_CHANNEL
        abfs = nn.ModuleList()
        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                ABF(
                    in_channel,
                    mid_channel,
                    out_channels[idx],
                    idx < len(in_channels) - 1,
                )
            )
        self.abfs = abfs[::-1]

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.abfs.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.abfs.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)

        index = kwargs['index']
        epc = kwargs['epc']
        ten = kwargs['ten']
        ind_ = list(index)
        ind = []
        for i in ind_:
            ind.append(int(i))
        loss_student = ce_loss(logits_student, target)
        a = 1
        b = 5*min(kwargs["epc"] / 20, 2)


        logprobs = F.log_softmax(logits_student, dim=1)
        tmp_target = target.view(-1, 1)
        logpt = logprobs.gather(1, tmp_target)
        logpt = logpt.view(-1)
        pt_s = Variable(logpt.data.exp())

        # if epc < 25:
        #     ce = CELoss(label_smooth=0.3, class_num=100)
        #     loss_ce_ = ce(logits_student, target)
        # elif epc >= 25 and epc < 50:
        #     ce = CELoss(label_smooth=0.25, class_num=100)
        #     loss_ce_ = ce(logits_student, target)
        # elif epc >= 50 and epc < 75:
        #     ce = CELoss(label_smooth=0.2, class_num=100)
        #     loss_ce_ = ce(logits_student, target)
        # elif epc >= 75 and epc < 100:
        #     ce = CELoss(label_smooth=0.1, class_num=100)
        #     loss_ce_ = ce(logits_student, target)
        # elif epc >= 100 and epc < 125:
        #     ce = CELoss(label_smooth=0.05, class_num=100)
        #     loss_ce_ = ce(logits_student, target)
        # else:
        #     loss_ce_ = F.cross_entropy(logits_student, target)

        # get features
        if self.stu_preact:
            x = features_student["preact_feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        else:
            x = features_student["feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        x = x[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(
            x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]
        ):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)
        features_teacher = features_teacher["preact_feats"][1:] + [
            features_teacher["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]

        # losses
        loss_kd, l_ce,loss_all = kd_loss(
            logits_student, logits_teacher, results,features_teacher,ind, epc,target,ten,self.T_MAX,self.T_MIN,self.Reduce,pt_s
        )
        if epc in [1,25,50,75,100,125]:
            #l_kd_ = list(l_kd_each)
            for i in range(len(loss_student)):
                ten[ind[i],0] = loss_student[i] #self.ce_loss_weight*loss_student[i]+self.kd_loss_weight*float(l_kd_[i])
                ten[ind[i],1] = 1-pt_s[i]

        tmp = 5 * min(kwargs["epoch"] / self.warmup_epochs, 2.0)*loss_all
        losses_dict = {
            "loss_ce": a*l_ce,
            "loss_kd": b*loss_kd,
            "loss_rekd":7
            * min(kwargs["epoch"] / self.warmup_epochs, 2.0)
            *loss_all
        }
        return logits_student, losses_dict


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (h,w), mode="nearest")#(shape, shape)
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        # output
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x
