#!/usr/bin/env python
# coding=utf-8
import datasets
import os
import logging
from tqdm.auto import tqdm
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
print(os.path.dirname(__file__))
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
import transformers
# W&B


from utils.arguments import get_training_arguments
from utils.model_utils import load_model, logger as model_logger, prepare_scheduler
from utils.data_utils import (
    load_dataset_splits,
    DataCollatorForSeq2Seq,
    preprocess_features_function
)
from utils.checkpoint_utils import maybe_save_checkpoint
from utils.metrics import calculate_accuracy, calculate_accuracy_perClass


logger = logging.getLogger(__name__)


def train_main():
    args = get_training_arguments()
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'logs.log'),
        filemode='w',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True
    )
    logger.info(args)


    # Initialize Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    device = accelerator.device
    print('current device:', device)
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    # Set random seed if specified
    if args.seed is not None:
        set_seed(args.seed)

    # 1) Load raw datasets
    raw_datasets = load_dataset_splits(args)

    # 2) Load model and tokenizer
    model, tokenizer = load_model(args)
    model_logger.info("Model & tokenizer loaded successfully.")

    # 3) Possibly slice dataset
    train_dataset = None

    if 'train' not in raw_datasets:
        raise ValueError(f"--It requires column \"train\" in raw_datasets!")
    train_dataset = raw_datasets['train']
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))

    # 4) Preprocess (tokenize) data
    if train_dataset is not None:
        train_dataset = train_dataset.map(
            preprocess_features_function,
            fn_kwargs={'args': args, 'raw_datasets': raw_datasets, 'tokenizer': tokenizer},
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset"
        )


    # 5) Dataloaders
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=(-100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id),
    )

    if train_dataset:
        train_dataset = train_dataset.remove_columns(["example_id", "offset_mapping"])
        
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
            shuffle=True
        )
    else:
        train_dataloader = None



    # 6) Optimizer
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=args.learning_rate)

    # Prepare with Accelerator
    model, optimizer = accelerator.prepare(model, optimizer)
    if train_dataloader:
        train_dataloader = accelerator.prepare(train_dataloader)

    # 7) Scheduler
    if train_dataloader:
        args.max_train_steps, args.num_train_epochs, lr_scheduler = prepare_scheduler(
            args, accelerator, train_dataloader, optimizer, args.max_train_steps, args.num_train_epochs
        )
    else:
        lr_scheduler = None

    # 8) Training Loop
    if train_dataloader:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        for epoch in range(args.num_train_epochs):
            model.train()
            total_loss = 0.0

            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

            
            avg_loss = total_loss / (step + 1)
            logger.info(f"Epoch {epoch} | Average loss = {avg_loss:.4f}")



            # If user wants checkpointing per epoch
            if epoch % args.checkpointing_num == 0 and epoch != 0:
                maybe_save_checkpoint(args, accelerator, model, tokenizer, epoch)


        if args.store_the_final_checkpoint:
            if accelerator.is_main_process:
                final_dir = os.path.join(args.output_dir, "final")
                logger.info(f"Saving final model to {final_dir}")
                model.save_pretrained(final_dir)
                tokenizer.save_pretrained(final_dir)

if __name__ == "__main__":
    train_main()
