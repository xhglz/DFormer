import argparse
import copy
import datetime
import os
import pprint
import random
import time
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from val_mm import evaluate, evaluate_msf

from models.builder import EncoderDecoder as segmodel
from models.distiller import RGBDDistiller
from utils.dataloader.dataloader import get_train_loader, get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.init_func import group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.pyt_utils import all_reduce_tensor, load_model


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="distill config file path")
parser.add_argument("--gpus", default=2, type=int, help="used gpu number")
parser.add_argument("-v", "--verbose", default=False, action="store_true")
parser.add_argument("--epochs", default=0)
parser.add_argument("--show_image", "-s", default=False, action="store_true")
parser.add_argument("--save_path", default=None)
parser.add_argument("--checkpoint_dir")
parser.add_argument("--continue_fpath")
parser.add_argument("--sliding", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile_mode", default="default")
parser.add_argument("--syncbn", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--mst", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--val_amp", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--pad_SUNRGBD", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--use_seed", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--local-rank", default=0)

torch.set_float32_matmul_precision("high")
import torch._dynamo

torch._dynamo.config.suppress_errors = True


def is_eval(epoch, config):
    ret = False
    if (epoch > int(config.checkpoint_start_epoch)):
        if (epoch % config.checkpoint_step == 0):
            ret = True
    return ret


class gpu_timer:
    def __init__(self, beta=0.6) -> None:
        self.start_time = None
        self.stop_time = None
        self.mean_time = None
        self.beta = beta
        self.first_call = True

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            return
        torch.cuda.synchronize()
        self.stop_time = time.perf_counter()
        elapsed = self.stop_time - self.start_time
        self.start_time = None
        if self.first_call:
            self.mean_time = elapsed
            self.first_call = False
        else:
            self.mean_time = self.beta * self.mean_time + (1 - self.beta) * elapsed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_teacher_student_weights(student, teacher, config, logger):
    teacher_weight = getattr(config, "teacher_weight", None)
    if teacher_weight is None or not os.path.exists(teacher_weight):
        raise FileNotFoundError(f"teacher_weight is missing or does not exist: {teacher_weight}")

    logger.info(f"load teacher weight from {teacher_weight}")
    load_model(teacher, teacher_weight, is_restore=False)

    student_weight = getattr(config, "student_weight", None)
    if student_weight is not None:
        if not os.path.exists(student_weight):
            raise FileNotFoundError(f"student_weight does not exist: {student_weight}")
        logger.info(f"load student weight from {student_weight}")
        load_model(student, student_weight, is_restore=False)
    else:
        logger.info("student_weight is None, using student config initialization only")


with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    config = copy.deepcopy(getattr(import_module(args.config), "C"))
    logger = get_logger(config.log_dir, config.log_file, rank=engine.local_rank)

    if args.compile:
        logger.warning("torch.compile is disabled in distill training to keep teacher/student path stable")
        args.compile = False

    if args.pad_SUNRGBD and config.dataset_name != "SUNRGBD":
        args.pad_SUNRGBD = False
        logger.warning("pad_SUNRGBD is only used for SUNRGBD dataset")
    if (args.pad_SUNRGBD) and (not config.backbone.startswith("DFormerv2")):
        raise ValueError("DFormerv1 is not recommended with pad_SUNRGBD")
    if (not args.pad_SUNRGBD) and config.backbone.startswith("DFormerv2") and config.dataset_name == "SUNRGBD":
        raise ValueError("DFormerv2 is not recommended without pad_SUNRGBD")
    config.pad = args.pad_SUNRGBD

    if args.use_seed:
        set_seed(config.seed)
        logger.info(f"set seed {config.seed}")
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        logger.info("use random seed")

    train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)
    val_loader, val_sampler = get_val_loader(
        engine,
        RGBXDataset,
        config,
        val_batch_size=int(config.batch_size) if config.dataset_name != "SUNRGBD" else int(args.gpus),
    )
    logger.info(f"val dataset len:{len(val_loader) * int(args.gpus)}")

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + "/{}".format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + "/tb"
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)
        pp = pprint.PrettyPrinter(indent=4)
        logger.info("config: \n" + pp.pformat(config))

    logger.info("args parsed:")
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))

    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)
    if args.syncbn:
        BatchNorm2d = nn.SyncBatchNorm
        logger.info("using syncbn")
    else:
        BatchNorm2d = nn.BatchNorm2d
        logger.info("using regular bn")

    teacher_cfg = copy.deepcopy(getattr(import_module(config.teacher_config), "C"))

    student = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        syncbn=args.syncbn,
    )
    teacher = segmodel(
        cfg=teacher_cfg,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        syncbn=args.syncbn,
    )
    load_teacher_student_weights(student, teacher, config, logger)

    model = RGBDDistiller(
        student=student,
        teacher=teacher,
        criterion=criterion,
        background=config.background,
        kd_weight=config.kd_weight,
        feat_weight=config.feat_weight,
        temperature=config.kd_temperature,
        feat_indices=config.feat_indices,
        student_feat_channels=getattr(config, "student_feat_channels", None),
        teacher_feat_channels=getattr(config, "teacher_feat_channels", None),
    )

    base_lr = config.lr
    params_list = []
    params_list = group_weight(params_list, model.student, BatchNorm2d, base_lr)
    if hasattr(model, "adapters") and len(getattr(model, "adapters")) > 0:
        params_list = group_weight(params_list, model.adapters, BatchNorm2d, base_lr)

    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params_list,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "SGDM":
        optimizer = torch.optim.SGD(
            params_list,
            lr=base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError

    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(
        base_lr,
        config.lr_power,
        total_iteration,
        config.niters_per_epoch * config.warm_up_epoch,
    )

    if engine.distributed:
        logger.info(".............distributed distill training.............")
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(
                model,
                device_ids=[engine.local_rank],
                output_device=engine.local_rank,
                find_unused_parameters=False,
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    logger.info("begin distill trainning:")

    train_timer = gpu_timer()
    eval_timer = gpu_timer()
    miou, best_miou = 0.0, 0.0
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    for epoch in range(engine.state.epoch, config.nepochs + 1):
        model.train()
        if engine.distributed:
            train_sampler.set_epoch(epoch)

        dataloader = iter(train_loader)
        sum_loss = 0.0
        sum_seg_loss = 0.0
        sum_kd_loss = 0.0
        sum_feat_loss = 0.0
        train_timer.start()

        for idx in range(config.niters_per_epoch):
            engine.update_iteration(epoch, idx)
            minibatch = next(dataloader)
            imgs = minibatch["data"].cuda(non_blocking=True)
            gts = minibatch["label"].cuda(non_blocking=True)
            modal_xs = minibatch["modal_x"].cuda(non_blocking=True)

            if args.amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss_dict = model(imgs, modal_xs, gts)
                    loss = loss_dict["loss"]
            else:
                loss_dict = model(imgs, modal_xs, gts)
                loss = loss_dict["loss"]

            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss.detach(), world_size=engine.world_size)
                reduce_seg_loss = all_reduce_tensor(loss_dict["seg_loss"], world_size=engine.world_size)
                reduce_kd_loss = all_reduce_tensor(loss_dict["kd_loss"], world_size=engine.world_size)
                reduce_feat_loss = all_reduce_tensor(loss_dict["feat_loss"], world_size=engine.world_size)
            else:
                reduce_loss = loss.detach()
                reduce_seg_loss = loss_dict["seg_loss"]
                reduce_kd_loss = loss_dict["kd_loss"]
                reduce_feat_loss = loss_dict["feat_loss"]

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]["lr"] = lr

            sum_loss += reduce_loss.item()
            sum_seg_loss += reduce_seg_loss.item()
            sum_kd_loss += reduce_kd_loss.item()
            sum_feat_loss += reduce_feat_loss.item()
            print_str = (
                f"Epoch {epoch}/{config.nepochs} "
                + f"Iter {idx + 1}/{config.niters_per_epoch}: "
                + f"lr={lr:.4e} "
                + f"loss={reduce_loss.item():.4f} "
                + f"seg={reduce_seg_loss.item():.4f} "
                + f"kd={reduce_kd_loss.item():.4f} "
                + f"feat={reduce_feat_loss.item():.4f} "
                + f"avg_loss={(sum_loss / (idx + 1)):.4f}"
            )

            if ((idx + 1) % int((config.niters_per_epoch) * 0.1) == 0 or idx == 0) and (
                (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed)
            ):
                print(print_str)

        logger.info(print_str)
        train_timer.stop()

        if (engine.distributed and engine.local_rank == 0) or (not engine.distributed):
            tb.add_scalar("train/total_loss", sum_loss / config.niters_per_epoch, epoch)
            tb.add_scalar("train/seg_loss", sum_seg_loss / config.niters_per_epoch, epoch)
            tb.add_scalar("train/kd_loss", sum_kd_loss / config.niters_per_epoch, epoch)
            tb.add_scalar("train/feat_loss", sum_feat_loss / config.niters_per_epoch, epoch)

        if is_eval(epoch, config):
            eval_timer.start()
            torch.cuda.empty_cache()
            if engine.distributed:
                with torch.no_grad():
                    model.eval()
                    device = torch.device("cuda")
                    if args.val_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            if args.mst:
                                all_metrics = evaluate_msf(
                                    model,
                                    val_loader,
                                    config,
                                    device,
                                    [0.5, 0.75, 1.0, 1.25, 1.5],
                                    True,
                                    engine,
                                    sliding=args.sliding,
                                )
                            else:
                                all_metrics = evaluate(model, val_loader, config, device, engine, sliding=args.sliding)
                    else:
                        if args.mst:
                            all_metrics = evaluate_msf(
                                model,
                                val_loader,
                                config,
                                device,
                                [0.5, 0.75, 1.0, 1.25, 1.5],
                                True,
                                engine,
                                sliding=args.sliding,
                            )
                        else:
                            all_metrics = evaluate(model, val_loader, config, device, engine, sliding=args.sliding)
                    if engine.local_rank == 0:
                        metric = all_metrics[0]
                        for other_metric in all_metrics[1:]:
                            metric.update_hist(other_metric.hist)
                        ious, miou = metric.compute_iou()
                        if miou > best_miou:
                            best_miou = miou
                            engine.save_and_link_checkpoint(
                                config.log_dir,
                                config.log_dir,
                                config.log_dir_link,
                                infor="_miou_" + str(miou),
                                metric=miou,
                            )
                        print("miou", miou, "best", best_miou)
            else:
                with torch.no_grad():
                    model.eval()
                    device = torch.device("cuda")
                    if args.val_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            if args.mst:
                                metric = evaluate_msf(
                                    model,
                                    val_loader,
                                    config,
                                    device,
                                    [0.5, 0.75, 1.0, 1.25, 1.5],
                                    True,
                                    engine,
                                    sliding=args.sliding,
                                )
                            else:
                                metric = evaluate(model, val_loader, config, device, engine, sliding=args.sliding)
                    else:
                        if args.mst:
                            metric = evaluate_msf(
                                model,
                                val_loader,
                                config,
                                device,
                                [0.5, 0.75, 1.0, 1.25, 1.5],
                                True,
                                engine,
                                sliding=args.sliding,
                            )
                        else:
                            metric = evaluate(model, val_loader, config, device, engine, sliding=args.sliding)
                    ious, miou = metric.compute_iou()
                if miou > best_miou:
                    best_miou = miou
                    engine.save_and_link_checkpoint(
                        config.log_dir,
                        config.log_dir,
                        config.log_dir_link,
                        infor="_miou_" + str(miou),
                        metric=miou,
                    )
                print("miou", miou, "best", best_miou)

            logger.info(f"Epoch {epoch} validation result: mIoU {miou}, best mIoU {best_miou}")
            if (engine.distributed and engine.local_rank == 0) or (not engine.distributed):
                tb.add_scalar("val/mIoU", miou, epoch)
            eval_timer.stop()

        eval_count = 0
        for i in range(engine.state.epoch + 1, config.nepochs + 1):
            if is_eval(i, config):
                eval_count += 1
        left_time = (train_timer.mean_time or 0) * (config.nepochs - engine.state.epoch) + (eval_timer.mean_time or 0) * eval_count
        eta = (datetime.datetime.now() + datetime.timedelta(seconds=left_time)).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"Avg train time: {(train_timer.mean_time or 0):.2f}s, avg eval time: {(eval_timer.mean_time or 0):.2f}s, left eval count: {eval_count}, ETA: {eta}"
        )
