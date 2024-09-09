import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def temp(logit):
    bs,C = logit.shape[0],logit.shape[1]
    # print(logit.norm(dim=1).mean(),logit.std(dim=1).mean())
    return (logit.std(dim=1)).reshape(bs,1)

# def kd_loss(logits_student, logits_teacher, temperature):
#     temp_t = temp(logits_teacher)
#     temp_s = temp(logits_student)
#     # print(temp_t,temp_s)
#     log_pred_student = F.log_softmax(logits_student, dim=1)
#     pred_teacher = F.softmax(logits_teacher, dim=1)
#     loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)#.mean()
#     # loss_kd *= temperature**2#(temp_s.view(-1)+temperature)*(temp_t.view(-1)+temperature)#(temp_s.view(-1)*temp_t.view(-1))
#     loss_kd *= (1+temp_s).view(-1)
#     return loss_kd.mean()


def kd_loss_clip(logits_student,temperature,clip_text,feat_stu,target,clip_feature,clip_logit):

    feat_stu = feat_stu/feat_stu.norm(dim=-1, keepdim=True)
    logit_feat = (100.0 * feat_stu.to(torch.float32)  @ clip_text.to(torch.float32)).cuda()
    
    clip_feature = clip_feature/clip_feature.norm(dim=-1, keepdim=True)
    logit_clip = (100.0 * clip_feature.to(torch.float32)  @ clip_text.to(torch.float32)).cuda()
    
    # gt_mask = _get_gt_mask(logit_feat, target)
    # logits_teacher = F.log_softmax(logit_feat - 1000.0 * gt_mask, dim=1)
    # log_pred_student_part2 = F.log_softmax(
    #     logits_student - 1000.0 * gt_mask, dim=1
    # )
    # loss_ce = 1.5 * F.cross_entropy(logit_feat, target)
    
 
    
    log_pred_student = F.softmax(logit_feat, dim=1)
    pred_teacher = F.softmax(logit_clip, dim=1)
    # # loss_kd1 = F.cross_entropy(log_pred_student, target)
    loss_kd1 = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    return loss_kd1


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.zero_shot_weight = torch.load("text_encode.pth")
        self.clip_logit = torch.load("logitregression.pth")#.cpu().numpy()
        self.clip_feat = torch.load('res_feature.pth').cpu().numpy()
        self.trans = transfor(256,512,768)
        # self.linear1 = torch.nn.Linear(256, 768)
    def get_learnable_parameters(self):
        return super().get_learnable_parameters()+list(self.trans.parameters())
    def forward_train(self, image, target, **kwargs):
                
        logits_student, feats = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        
        
        zeroshot_weights = self.zero_shot_weight
        epoch = kwargs['epoch']
        # logits_clip = kwargs['clip_logit']
        index = kwargs['index']
        idx = []
        for i in list(index):
            idx.append(int(i))
        clip_out = self.clip_logit
        clip_logit = torch.from_numpy(clip_out[idx]).to(torch.float32).cuda()
        clip_feat = torch.from_numpy(self.clip_feat[idx]).to(torch.float32).cuda()
        feat_s = self.trans(feats['pooled_feat'].unsqueeze(-1).unsqueeze(-1))
        feat_final = feat_s.squeeze(-1).squeeze(-1)
        # feat_final = self.linear1(feats['pooled_feat'])
        feat_final = feat_final/feat_final.norm(dim=-1, keepdim=True)
        # logit_feat = (100.0 * feat_final.to(torch.float32)  @ zeroshot_weights.to(torch.float32)).cuda()
        # clip_logit = (100.0 * clip_feat.to(torch.float32)  @ zeroshot_weights.to(torch.float32)).cuda()
        # print(clip_logit)
        # losses
        loss_ce = 0.5 * F.cross_entropy(logits_student, target)
        loss_kd = 1 * kd_loss_clip(
            logits_student,4,zeroshot_weights,feat_final,target,clip_feat,clip_logit
        )
        # res = accuracy(clip_logit, target)
        # print(res)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
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
    
