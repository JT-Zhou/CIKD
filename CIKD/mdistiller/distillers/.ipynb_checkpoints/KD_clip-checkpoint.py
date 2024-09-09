import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit) / (1e-7 + stdv)

def logit_tg(logit,target):
    logprobs = F.log_softmax(logit, dim=1)
    tmp_target = target.view(-1, 1)
    logpt = logprobs.gather(1, tmp_target)
    logpt = logpt.view(-1)
    pt_s = Variable(logpt.data.exp())
    return pt_s

def kd_loss_clip(logits_student, temperature, clip_text, feat_stu, feat_tea, target, clip_feature, epoch):
    clip_text = clip_text.to(device, torch.float32)
    feat_stu = feat_stu.to(device, torch.float32)
    clip_feature = clip_feature.to(device, torch.float32)
    feat_stu = feat_stu / feat_stu.norm(dim=-1, keepdim=True)
    feat_tea = feat_tea / feat_tea.norm(dim=-1, keepdim=True)
    logit_stu = (100.0 * feat_stu @ clip_text)  # .cuda().to(torch.float32).to(torch.float32)
    logit_tea = (100.0 * feat_tea @ clip_text)

    logit_clip = (
                100.0 * clip_feature @ clip_text)  # .cuda().to(torch.float32).to(torch.float32)[correct_clip[0]][correct_clip[0]]

    _, pred_clip = logit_clip.topk(1, 1, True, True)
    # _, pred = output.topk(maxk, 1, True, True)
    pred_clip = pred_clip.t()
    correct_clip = pred_clip.eq(target.reshape(1, -1).expand_as(pred_clip))

    loss_kd1 = kd_loss(
        logit_stu[correct_clip[0]], logit_clip[correct_clip[0]], 2
    )
    loss_kd3 = kd_loss(
        logit_stu, logit_tea, 2
    )

    return loss_kd1 + 0.5 * loss_kd3  # 3*(epoch/10)0.5*(epoch/10)*loss_kd2+


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2  # (temp_s.view(-1)+temperature)*(temp_t.view(-1)+temperature)#(temp_s.view(-1)*temp_t.view(-1))
    return loss_kd



def kd_loss_sp(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)#.mean()
    loss_kd *= temperature ** 2  # (temp_s.view(-1)+temperature)*(temp_t.view(-1)+temperature)#(temp_s.view(-1)*temp_t.view(-1))
    return loss_kd

def kd_loss_sp2(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / (temperature-1), dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)#.mean()
    # loss_kd *= temperature ** 2  # (temp_s.view(-1)+temperature)*(temp_t.view(-1)+temperature)#(temp_s.view(-1)*temp_t.view(-1))
    loss_kd = loss_kd*(temperature*(temperature-1))
    return loss_kd

def kd_loss_sp3(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / (temperature), dim=1)
    pred_teacher = F.softmax(logits_teacher / (temperature-1), dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)#.mean()
    # loss_kd *= temperature ** 2  # (temp_s.view(-1)+temperature)*(temp_t.view(-1)+temperature)#(temp_s.view(-1)*temp_t.view(-1))
    loss_kd = loss_kd*(temperature*(temperature-1))
    return loss_kd


class KD_clip(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD_clip, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.trans = MLP(512, 512, 768)
        # self.trans_t = MLP(256, 512, 768)
        self.zero_shot_weight = torch.load("text.pth")
        self.clip_logit = torch.load("clip_img_logits.pth").cpu().numpy()
        self.clip_feat = torch.load('clip_img_feats.pth').cpu().numpy()

    def get_learnable_parameters(self):
        return super().get_learnable_parameters()# + list(self.trans.parameters())# + list(self.trans_t.parameters())

    def forward_train(self, image, target, **kwargs):
        with torch.no_grad():
            logits_teacher, feats_t = self.teacher(image)
        # logits_teacher=0
        logits_student, feats = self.student(image)
        zeroshot_weights = self.zero_shot_weight
        epoch = kwargs['epoch']
        logits_clip = self.clip_logit
        index = kwargs['index']
        idx = []
        for i in list(index):
            idx.append(int(i))
        feat_clip = self.clip_feat
        logits_clip_batch = torch.from_numpy(logits_clip[idx]).to(torch.float32).cuda()
        
        tmp1 = F.softmax(logits_clip_batch, dim=1)
        tmp3 = F.softmax(logits_student, dim=1)
        _, pred_clip = F.softmax(0.5*tmp1+min(epoch/20,1)*tmp3, dim=1).topk(1, 1, True, True)
        pred_clip = pred_clip.t()
        correct_clip = pred_clip.eq(target.reshape(1, -1).expand_as(pred_clip))[0]
        uncorrect_clip = ~correct_clip

        loss_ce = 0.1 * F.cross_entropy(logits_student, target)  # .half()

        loss_kd1 = kd_loss_sp(
            logits_student[correct_clip], logits_teacher[correct_clip], 4
        )  # .half(), temp_t1, temp_s1

        loss_kd2 =  kd_loss_sp(
            logits_student[uncorrect_clip], logits_teacher[uncorrect_clip], 4
        )  # .half(),temp_t2, temp_s2
        loss_kd = torch.cat([0.9 * loss_kd1, 1.2 * loss_kd2])
        loss_kd = loss_kd.mean()

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd#s+loss_kd2
        }
        # print(loss_kd)
        return logits_student, losses_dict


class transfor(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(transfor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            # nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            # nn.BatchNorm2d(out_channel),
        )
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        # output
        # if x.shape[-1] != out_shape:
        #     x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y


class MLP(torch.nn.Module):

    def __init__(self, in_channel, mid_channel, out_channel):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(in_channel, mid_channel)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(mid_channel, out_channel)  # 2个隐层

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count