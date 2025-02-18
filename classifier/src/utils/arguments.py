# utils/arguments.py
import argparse

def get_training_arguments():
    parser = argparse.ArgumentParser(description="Train a Seq2Seq QA/classification model.")

    # Original arguments from your snippet or prior code


    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--ignore_pad_token_for_loss", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--source_prefix", type=str, default=None)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)



    parser.add_argument("--overwrite_cache", action="store_true")

    parser.add_argument("--max_answer_length", type=int, default=30)
    parser.add_argument("--val_max_answer_length", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)

    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--question_column", type=str, default="question")
    parser.add_argument("--answer_column", type=str, default="answer")

    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        choices=["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--checkpointing_num", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--store_the_final_checkpoint",  action="store_true")
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--report_to", type=str, default="all")

    parser.add_argument("--doc_stride", type=int, default=128)


    args = parser.parse_args()
    return args

def get_evaluation_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a Seq2Seq QA/classification model.")

    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--ignore_pad_token_for_loss", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--max_answer_length", type=int, default=30)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--question_column", type=str, default="question")
    parser.add_argument("--answer_column", type=str, default="answer")
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--pad_to_max_length", action="store_true")

    args = parser.parse_args()
    return args

def get_merge_arguments():
    parser = argparse.ArgumentParser(description="Merge two model checkpoints by averaging weights.")
    parser.add_argument("--model_path_1", type=str, required=True)
    parser.add_argument("--model_path_2", type=str, required=True)
    parser.add_argument("--merged_output_path", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    # We'll also accept some standard model-loading args for compatibility:
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--model_type", type=str, default=None)

    args = parser.parse_args()
    return args
