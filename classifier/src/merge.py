#!/usr/bin/env python
# coding=utf-8

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import logging
import torch

from transformers import AutoModelForSeq2SeqLM

from utils.arguments import get_merge_arguments
from utils.model_utils import load_model, tokenizer_from_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_main():
    args = get_merge_arguments()

    # Load model 1
    logger.info(f"Loading first model from {args.model_path_1}")
    args.model_name_or_path = args.model_path_1
    model1, tokenizer1 = load_model(args)

    # Load model 2
    logger.info(f"Loading second model from {args.model_path_2}")
    args.model_name_or_path = args.model_path_2
    model2, tokenizer2 = load_model(args)

    # Weighted average
    logger.info(f"Merging with alpha={args.alpha}. Output => {args.merged_output_path}")

    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    merged_sd = {}
    for key in state_dict1.keys():
        merged_sd[key] = args.alpha * state_dict1[key] + (1 - args.alpha) * state_dict2[key]

    # Create new model with merged weights
    merged_model = AutoModelForSeq2SeqLM.from_config(model1.config)
    merged_model.load_state_dict(merged_sd)

    # Save
    os.makedirs(args.merged_output_path, exist_ok=True)
    merged_model.save_pretrained(args.merged_output_path)
    tokenizer1.save_pretrained(args.merged_output_path)

    logger.info("Merged model saved successfully.")


if __name__ == "__main__":
    merge_main()
