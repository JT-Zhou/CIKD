import os
import argparse
import torch
import json
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict,trainer_ts_dict


def main(cfg, resume, opts,Cuda_id):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    train_loader, val_loader, num_data, num_classes,num_test = get_dataset(cfg)

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )

        elif cfg.DISTILLER.TYPE=='IAKD':
            dim = []
            dim.append(cfg.IAKD.SHAPES)#s
            dim.append(cfg.IAKD.OUT_SHAPES)#t
            dim.append(cfg.IAKD.IN_CHANNELS)#s
            dim.append(cfg.IAKD.OUT_CHANNELS)#t

            # dim.append([1,8,16,32])#s
            # dim.append([1,8,16,32])#t
            # dim.append([64,128,256,256])#s
            # dim.append([64,128,256,256])#t

            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg,dim
            )
        elif cfg.DISTILLER.TYPE=='KD_M' or cfg.DISTILLER.TYPE=='DKD_M':
            net2, pretrain_model_path2 = cifar_model_dict[cfg.DISTILLER.TEACHER]
            model_teacher2 = net(num_classes=num_classes)
            model_teacher2.load_state_dict(load_checkpoint(pretrain_model_path2)["model"])
            
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, model_teacher2,cfg
            )
            
        else:
            tmp = cfg.DISTILLER.TYPE
            di = distiller_dict
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )

    distiller = torch.nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    #distiller = model_student
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default=r"G:\Python Prog\github\mdistiller-master\configs\cifar100\IAKD\res32x4_8x4.yaml")#dkd\res56x4_res20x4.yaml
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--cuda", type=str, default=r"cuda:0")
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.resume, args.opts,args.cuda)
