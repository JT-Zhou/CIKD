import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ._base import Distiller


def kd_loss(logits_student, logits_teacher, temperature):
    a = temperature.shape[0]
    temperature = temperature.reshape(a,1)
    logits_teacher = torch.div(logits_teacher,temperature)
    logits_student = torch.div(logits_student,temperature)
    log_pred_student = F.log_softmax(logits_student, dim=1)
    pred_teacher = F.softmax(logits_teacher, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    # loss_kd *= temperature**2
    loss_kd = torch.mul(loss_kd,temperature**2)
    return loss_kd


class CTKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(CTKD, self).__init__(student, teacher)
        # self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.CTKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.CTKD.LOSS.KD_WEIGHT

        self.cos_value = cfg.CTKD.cosine_decay
        self.decay_max = cfg.CTKD.decay_max
        self.decay_min = cfg.CTKD.decay_min
        self.decay_loops = cfg.CTKD.decay_loops
        self.T_MAX = cfg.CTKD.T_MAX
        self.T_MIN = cfg.CTKD.T_MIN

        if self.cos_value:
            self.gradient_decay = CosineDecay(max_value=self.decay_max, min_value=self.decay_min, num_loops=self.decay_loops)
        else:
            self.gradient_decay = LinearDecay(max_value=self.decay_max, min_value=self.decay_min, num_loops=self.decay_loops)
        self.mlp = Global_T()

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.mlp.parameters())

    # def get_learnable_parameters(self):
    #     num_p = 0
    #     for p in self.mlp.parameters():
    #         num_p += p.numel()
    #     return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        epoch = kwargs['epoch']
        decay_value = 1

        decay_value = self.gradient_decay.get_value(epoch)
        temp = self.mlp(logits_teacher, logits_student, decay_value)  # (teacher_output, student_output)
        temp = self.T_MIN + self.T_MAX * torch.sigmoid(temp)
        temp = temp.cuda()
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, temp
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict


class Global_T(nn.Module):
    def __init__(self):
        super(Global_T, self).__init__()

        self.global_T = nn.Parameter(torch.ones(1), requires_grad=True)
        self.grl = GradientReversal()

    def forward(self, fake_input1, fake_input2, lambda_):
        return self.grl(self.global_T, lambda_)



class Instance_MLP(nn.Module):
    def __init__(self, num_classes=100, inter_channel=256, topk_num=100):
        super(Instance_MLP, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * topk_num, inter_channel, 1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channel, 1, 1)
        )
        self.grl = GradientReversal()

        self.topk_num = topk_num

        for m in self.conv2:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight.data, 0)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, teacher_out, student_out, lambda_):
        stu_out_topk = torch.zeros(student_out.shape[0], self.topk_num)

        tea_out = F.softmax(teacher_out, dim=1)
        stu_out = F.softmax(student_out, dim=1)

        tea_out_topk, out_index = tea_out.topk(self.topk_num, dim=1)
        stu_out_topk = stu_out.gather(1, out_index)

        comb_out = torch.cat([tea_out_topk, stu_out_topk], dim=1)
        comb_out = comb_out.view(comb_out.shape[0], comb_out.shape[1], 1, 1).detach()

        T = self.conv1(comb_out)
        T = self.conv2(T)
        T = self.grl(T, lambda_)
        T = T.view(-1)

        return T


from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()
        # self.lambda_ = lambda_

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)


class CosineDecay(object):
    def __init__(self,
                max_value,
                min_value,
                num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value


class LinearDecay(object):
    def __init__(self,
                max_value,
                min_value,
                num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops - 1

        value = (self._max_value - self._min_value) / self._num_loops
        value = i * (-value)

        return value