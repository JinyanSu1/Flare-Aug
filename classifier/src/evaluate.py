#!/usr/bin/env python
# coding=utf-8
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
import json
from utils.arguments import get_evaluation_arguments
from utils.model_utils import load_model
from utils.data_utils import DataCollatorForSeq2Seq, preprocess_features_function, load_dataset_splits
from utils.metrics import calculate_accuracy, calculate_accuracy_perClass

logger = logging.getLogger(__name__)

def evaluate_main():
    args = get_evaluation_arguments()
    logging.basicConfig(level=logging.INFO)

    # 1) Accelerator
    accelerator = Accelerator()

    # 2) Load model & tokenizer
    model, tokenizer = load_model(args)
    model = accelerator.prepare(model)

    # 3) Load the raw dataset splits
    raw_datasets = load_dataset_splits(args)
    if "validation" not in raw_datasets:
        raise ValueError("No validation set found in raw_datasets!")

    eval_examples = raw_datasets["validation"]
    if args.max_eval_samples is not None:
        eval_examples = eval_examples.select(range(args.max_eval_samples))

    # 4) Preprocess (tokenize) eval dataset
    eval_dataset = eval_examples.map(
        preprocess_features_function,
        fn_kwargs={'args': args, 'raw_datasets': raw_datasets, 'tokenizer': tokenizer},
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=eval_examples.column_names,
        desc="Tokenizing evaluation dataset"
    )

    # 5) Build dataloader
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=(-100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id),
        pad_to_multiple_of=8 if accelerator.mixed_precision == 'fp16' else None,
    )
    eval_dataset = eval_dataset.remove_columns(["example_id", "offset_mapping"])   
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    eval_dataloader = accelerator.prepare(eval_dataloader)

    # 6) Evaluate
    logger.info("Starting evaluation...")
    model.eval()

    # Example mapping for multiple-choice:
    # Modify these if you have more or fewer choices, or your data uses different tokens.
    label_to_option = {0: 'A', 1: 'B', 2: 'C'}

    # If your multiple-choice answers are more than one token, adjust this logic!
    a_token_id = tokenizer("A", add_special_tokens=False).input_ids[0]
    b_token_id = tokenizer("B", add_special_tokens=False).input_ids[0]
    c_token_id = tokenizer("C", add_special_tokens=False).input_ids[0]

    predictions = []
    eval_loss_sum = 0

    for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        with torch.no_grad():
            # Forward pass to get the loss (optional, if you want average eval loss)
            outputs = model(**batch)
            loss = outputs.loss
            eval_loss_sum += loss.item()

            # -- Borrowed logic: gather the distribution from `generate()` --
            # Make sure to specify `return_dict_in_generate=True, output_scores=True` 
            # so we can inspect the logits for multiple-choice.
            generation_out = accelerator.unwrap_model(model).generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=args.max_answer_length,
                return_dict_in_generate=True,
                output_scores=True,
            )

            # For single-step multiple-choice, the relevant logits are in `scores[0]`.
            # shape: [batch_size, vocab_size]
            scores = generation_out.scores[0]

            # Gather logits for 'A', 'B', 'C'
            stacked_scores = torch.stack([
                scores[:, a_token_id],
                scores[:, b_token_id],
                scores[:, c_token_id]
            ])  # shape: [3, batch_size]

            # Softmax across the 3 possible choices, then argmax
            probs = torch.nn.functional.softmax(stacked_scores, dim=0)  # [3, batch_size]
            predicted_indices = probs.argmax(dim=0).cpu().numpy()  # shape: [batch_size]
            preds = [label_to_option[idx] for idx in predicted_indices]

            # Convert gold labels from token IDs to 'A','B','C' etc.
            # This depends on how your data is structured. The example below:
            #   - The correct label is in batch["labels"], 
            #   - We assume each example has exactly one token that corresponds to 'A','B','C'.
            labels = batch["labels"]  # shape: [batch_size, seq_len]
            # We'll just look at the first token per row (or whichever is the "real" label).
            labels = accelerator.gather_for_metrics(labels)
            labels = labels.cpu().numpy()
            predictions = predictions + preds
            if args.ignore_pad_token_for_loss:
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    gold_answers = eval_examples['answer']

    dict_id_pred_results = {qid : {'prediction': pred, 'answer' : ans, 'dataset_name' : data} for qid, pred, ans, data in zip(eval_examples['id'], predictions, gold_answers, eval_examples['dataset_name'])}
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "dict_id_pred_results.json"), "w") as f:
        json.dump(dict_id_pred_results, f, indent=4)
    assert len(gold_answers) == len(predictions)
    final_acc_score = calculate_accuracy(gold_answers, predictions)
    final_eval_results = {'final_acc_score' : final_acc_score}

    logger.info(f"Evaluation metrics: {final_eval_results}")
    with open(os.path.join(args.output_dir, "final_eval_results.json"), "w") as f:
        json.dump(final_eval_results, f)
    final_eval_results_perClass = calculate_accuracy_perClass(gold_answers, predictions)

    logger.info(f"Evaluation metrics per class: {final_eval_results_perClass}")
    print(final_eval_results_perClass)

    with open(os.path.join(args.output_dir, "final_eval_results_perClass.json"), "w") as f:
        json.dump(final_eval_results_perClass, f, indent=4)




if __name__ == "__main__":
    evaluate_main()
