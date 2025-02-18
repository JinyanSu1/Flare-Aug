# utils/model_utils.py
import logging
import math

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_scheduler,
)

logger = logging.getLogger(__name__)


def load_model(args):
    """
    Incorporates your snippet's logic for loading config, tokenizer, model.
    """
    # 1) Load config
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # 2) Load tokenizer
    tokenizer = tokenizer_from_args(args)

    # 3) Load model
    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    return model, tokenizer


def tokenizer_from_args(args):
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "Use --tokenizer_name or --model_name_or_path."
        )
    return tokenizer


def prepare_scheduler(args, accelerator, dataloader, optimizer, max_train_steps, train_epoch):
    """
    Copied from your snippet. Creates and returns (max_train_steps, num_epochs, lr_scheduler).
    """
    overrode_max_train_steps = False

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)

    if max_train_steps is None:
        max_train_steps = train_epoch * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    lr_scheduler = accelerator.prepare(lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)

    if overrode_max_train_steps:
        max_train_steps = train_epoch * num_update_steps_per_epoch
    train_epoch = math.ceil(max_train_steps / num_update_steps_per_epoch)

    return max_train_steps, train_epoch, lr_scheduler
