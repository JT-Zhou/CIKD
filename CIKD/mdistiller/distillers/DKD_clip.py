import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def dkd_loss_sp(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(
                1)  # F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
            * (temperature ** 2)
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="none").sum(1)
            * (temperature ** 2)
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


class DKD_clip(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(DKD_clip, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP
        self.trans = MLP(512, 512, 768)
        # self.trans_t = MLP(256, 512, 768)
        self.zero_shot_weight = torch.load("text.pth")
        self.clip_logit = torch.load("clip_img_logits.pth").cpu().numpy()
        self.clip_feat = torch.load('clip_img_feats.pth').cpu().numpy()

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.trans.parameters())

    def forward_train(self, image, target, **kwargs):
        logits_student, feats = self.student(image)
        with torch.no_grad():
            logits_teacher, feats_t = self.teacher(image)

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

        feat_s = self.trans(feats['pooled_feat'])
        feat_final_stu = feat_s

        clip_text = zeroshot_weights.to(device, torch.float32)
        feat_stu = feat_final_stu.to(device, torch.float32)
        feat_stu = feat_stu / feat_stu.norm(dim=-1, keepdim=True)
        logit_stu = (100.0 * feat_stu @ clip_text)

        tmp1 = F.softmax(logits_clip_batch, dim=1)
        tmp2 = F.softmax(logits_student, dim=1)
        _, pred_clip = F.softmax(0.5*tmp1+min(epoch/20,1)*tmp2, dim=1).topk(1, 1, True, True)
        pred_clip = pred_clip.t()
        correct_clip = pred_clip.eq(target.reshape(1, -1).expand_as(pred_clip))[0]
        uncorrect_clip = ~correct_clip


        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd1 = dkd_loss_sp(
            logits_student[correct_clip],
            logits_teacher[correct_clip],
            target[correct_clip],
            self.alpha,
            self.beta,
            self.temperature,
        )
        loss_dkd2 = dkd_loss_sp(
            logits_student[uncorrect_clip],
            logits_teacher[uncorrect_clip],
            target[uncorrect_clip],
            self.alpha,
            self.beta,
            self.temperature,
        )
        loss_dkd = torch.cat([1 * loss_dkd1, 1.3 * loss_dkd2])
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * loss_dkd.mean()

        loss_kd_stu2clip = dkd_loss(
            logit_stu,
            logits_clip_batch,
            target,
            self.alpha,
            self.beta,
            2,
        )
        loss_kd_stu2clip_di = dkd_loss(
            logits_student,
            logits_clip_batch,
            target,
            self.alpha,
            self.beta,
            2,
        )

        loss_clip = min(epoch / 20, 0.4) * (loss_kd_stu2clip) + (0.1 * loss_kd_stu2clip_di)

        losses_dict = {
            "loss_ce": loss_ce,
            'loss_clip': loss_clip,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict


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