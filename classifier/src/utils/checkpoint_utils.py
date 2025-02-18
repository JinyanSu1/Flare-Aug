# utils/checkpoint_utils.py
import os
import logging

logger = logging.getLogger(__name__)

def maybe_save_checkpoint(args, accelerator, model, tokenizer, epoch):
    """
    Saves a checkpoint if we hit a checkpointing step or if `force_save=True`.
    `args.checkpointing_steps` can be:
        - None (never checkpoint except final),
        - "epoch" (checkpoint every epoch),
        - an integer N (checkpoint every N steps).
    """
    if not accelerator.is_main_process:
        return

    if args.checkpointing_num is None:
        return

    # If 'epoch' mode and we want forced save after each epoch

    output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving checkpoint at {output_dir}")
    accelerator.unwrap_model(model).save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return


