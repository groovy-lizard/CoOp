"""Main inference module for train and eval"""
import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from yacs.config import CfgNode as CN

import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.fairface
import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip


def print_args(args_to_print, cfg):
    """print arguments"""
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args_to_print.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        key_args = args_to_print.__dict__[key]
        print(f"{key}: {key_args}")
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, reset_args):
    """reset confs"""
    if reset_args.root:
        cfg.DATASET.ROOT = reset_args.root

    if reset_args.output_dir:
        cfg.OUTPUT_DIR = reset_args.output_dir

    if reset_args.resume:
        cfg.RESUME = reset_args.resume

    if reset_args.seed:
        cfg.SEED = reset_args.seed

    if reset_args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = reset_args.source_domains

    if reset_args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = reset_args.target_domains

    if reset_args.transforms:
        cfg.INPUT.TRANSFORMS = reset_args.transforms

    if reset_args.trainer:
        cfg.TRAINER.NAME = reset_args.trainer

    if reset_args.backbone:
        cfg.MODEL.BACKBONE.NAME = reset_args.backbone

    if reset_args.head:
        cfg.MODEL.HEAD.NAME = reset_args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    # 'middle' or 'end' or 'front'
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(setup_args):
    """Set up configurations"""
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if setup_args.dataset_config_file:
        cfg.merge_from_file(setup_args.dataset_config_file)

    # 2. From the method config file
    if setup_args.config_file:
        cfg.merge_from_file(setup_args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, setup_args)

    # 4. From optional input arguments
    cfg.merge_from_list(setup_args.opts)

    cfg.freeze()

    return cfg


def main(main_args):
    """Main function entrypoint"""
    cfg = setup_cfg(main_args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {cfg.SEED}")
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(main_args, cfg)
    print("Collecting env info ...")
    print(f"** System info **\n{collect_env_info()}\n")

    trainer = build_trainer(cfg)

    if main_args.eval_only:
        trainer.load_model(main_args.model_dir, epoch=main_args.load_epoch)
        trainer.test()
        return

    if not main_args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str,
                        default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1,
        help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+",
        help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+",
        help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+",
        help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="",
        help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="",
                        help="name of trainer")
    parser.add_argument("--backbone", type=str, default="",
                        help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true",
                        help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int,
        help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
