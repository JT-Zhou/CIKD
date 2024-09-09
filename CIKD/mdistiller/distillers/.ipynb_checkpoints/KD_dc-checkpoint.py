import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit) / (1e-7 + stdv)

def kd_loss_clip(logits_student,temperature,clip_text,feat_stu,target,clip_feature):
    clip_text = clip_text.to(device,torch.float32)
    feat_stu = feat_stu.to(device,torch.float32)
    clip_feature = clip_feature.to(device,torch.float32)
    feat_stu = feat_stu/feat_stu.norm(dim=-1, keepdim=True)
    logit_feat = (100.0 * feat_stu @ clip_text)#.cuda().to(torch.float32).to(torch.float32)
    
    clip_feature = clip_feature/clip_feature.norm(dim=-1, keepdim=True)
    logit_clip = (100.0 * clip_feature  @ clip_text)#.cuda().to(torch.float32).to(torch.float32)
    
    # gt_mask = _get_gt_mask(logit_feat, target) - mean
    # logits_teacher = F.log_softmax(logit_feat - 1000.0 * gt_mask, dim=1)
    # log_pred_student_part2 = F.log_softmax(
    #     logits_student - 1000.0 * gt_mask, dim=1
    # )
    # loss_ce = 1.5 * F.cross_entropy(logit_feat, target)
    
    temp_t = normalize(logit_clip)
    temp_s = normalize(logit_feat)
    log_pred_student = F.log_softmax(temp_s, dim=1)
    pred_teacher = F.softmax(temp_t, dim=1)
    # print(temp_t.mean(),temp_s.mean())
    # print(log_pred_student.topk(5))
    # print(pred_teacher.topk(5))
    # # loss_kd1 = F.cross_entropy(log_pred_student, target)
    loss_kd1 = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    return loss_kd1

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student, dim=1)
    pred_teacher = F.softmax(logits_teacher, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / 0.5 - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature*2)
        / target.shape[0]
    )
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

class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.trans = transfor(256,512,768)
        self.zero_shot_weight = torch.load("text_encode.pth")
        self.clip_logit = torch.load("logitregression.pth")#.cpu().numpy()
        self.clip_feat = torch.load('res_feature.pth').cpu().numpy()
    def get_learnable_parameters(self):
        return super().get_learnable_parameters()+ list(self.trans.parameters())


    # def get_extra_parameters(self):
    #     num_p = 0
    #     for p in self.trans.parameters():
    #         num_p += p.numel()
    #     return num_p

    def forward_train(self, image, target, **kwargs):
        # with torch.no_grad():
        #     logits_teacher, _ = self.teacher(image)
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
        clip_feature = torch.from_numpy(feat_clip[idx]).to(torch.float32).cuda()
        logits_clip_batch = torch.from_numpy(logits_clip[idx]).to(torch.float32).cuda()
        
        # feat_s = self.trans(feats['pooled_feat'].unsqueeze(-1).unsqueeze(-1))
        # feat_final = feat_s.squeeze(-1).squeeze(-1)
        
        
        # clip_text = zeroshot_weights.to(device,torch.float32)
        # feat_stu = feat_final.to(device,torch.float32)
        # clip_feature = clip_feature.to(device,torch.float32)
        # feat_stu = feat_stu/feat_stu.norm(dim=-1, keepdim=True)
        # logit_feat = (100.0 * feat_stu @ clip_text)#.cuda().to(torch.float32).to(torch.float32)
        # clip_feature = clip_feature/clip_feature.norm(dim=-1, keepdim=True)
        # logit_clip = (100.0 * clip_feature  @ clip_text)#.cuda().to(torch.float32).to(torch.float32)
        temp_t = normalize(logits_clip_batch)
        temp_s = normalize(logits_student)
        
        
        loss_ce = 1 * F.cross_entropy(logits_student, target)  # .half()
        loss_dkd = min(kwargs["epoch"] / 20, 1.0) * dkd_loss(
            temp_s,
            temp_t,
            target,
            1,
            8,
            4,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_clip": loss_dkd,
            # "loss_kd": loss_kd
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
    
    
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        # _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        tmp = correct[0]
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