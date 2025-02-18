# utils/data_utils.py

import logging
import datasets
import numpy as np
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq as HFDataCollator
import json, jsonlines
import os, sys
logger = logging.getLogger(__name__)


class DataCollatorForSeq2Seq(HFDataCollator):
    """
    Inherit from HF's DataCollatorForSeq2Seq, but you can add custom logic if needed.
    """
    pass


def load_dataset_splits(args):
    """
    Load 'train' and 'validation' splits from user-specified files or a dataset name.
    """
    data_files = {}
    if hasattr(args, 'train_file'):
        data_files["train"] = args.train_file
        file_path = args.train_file
    if hasattr(args, 'validation_file'):
        data_files["validation"] = args.validation_file
        file_path = args.validation_file
        # figure out extension
        
        
    extension = file_path.split(".")[-1]
    raw_datasets = datasets.load_dataset(extension, data_files=data_files)
    return raw_datasets


def preprocess_dataset(args, raw_datasets):
    """
    Your snippet function: checks if question_column and answer_column exist, returns them.
    """
    if "train" in raw_datasets:
        column_names = raw_datasets['train'].column_names
    if 'validation' in raw_datasets:
        column_names = raw_datasets['validation'].column_names
        
    

    question_column = args.question_column
    if question_column not in column_names:
        raise ValueError(
            f"--question_column value '{question_column}' not in: {', '.join(column_names)}"
        )

    answer_column = args.answer_column
    if answer_column not in column_names:
        raise ValueError(
            f"--answer_column value '{answer_column}' not in: {', '.join(column_names)}"
        )

    return question_column, answer_column


def preprocess_features_function(examples, args, raw_datasets, tokenizer):
    """
    Your snippet's tokenization logic for each batch (train or eval).
    - calls preprocess_dataset to get question_column, answer_column
    - tokenizes question, then tokenizes answers as labels
    """
    question_column, answer_column = preprocess_dataset(args, raw_datasets)

    max_answer_length = args.max_answer_length
    padding = "max_length" if args.pad_to_max_length else False
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # Strip question
    examples[question_column] = [q.strip() for q in examples[question_column]]

    # Tokenize questions
    model_inputs = tokenizer(
        examples[question_column],
        truncation=True,
        max_length=max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=padding,
    )

    # Tokenize answers
    targets = examples[answer_column]
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_answer_length,
            padding=padding,
            truncation=True
        )

    # If we're padding, replace pad_token_id with -100 if ignoring pad
    if padding == "max_length" and args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    # "overflow_to_sample_mapping" so each tokenized chunk can map back to the original example
    sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

    # Build final inputs
    model_inputs["example_id"] = []
    out_labels = []

    for i in range(len(model_inputs["input_ids"])):
        sample_index = sample_mapping[i]
        model_inputs["example_id"].append(examples["id"][sample_index])
        out_labels.append(labels["input_ids"][sample_index])

    model_inputs["labels"] = out_labels
    return model_inputs


def post_processing_function(tokenizer, args, raw_datasets, examples, features, outputs, stage="eval"):
    """
    Your snippet for decoding predicted tokens. 
    E.g. used in a real QA context for final text extraction. 
    Here it just decodes the logits -> text.
    """
    preds = outputs
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    return decoded_preds


def create_and_fill_np_array(all_gen_tokens, dataset, max_len):
    """
    Your snippet function: Create a large array of shape [len(dataset), max_len],
    fill it with -100, then copy each tensor's tokens into it.
    """
    step = 0
    gen_toks_concat = np.full((len(dataset), max_len), -100)

    for i, gen_tok in enumerate(all_gen_tokens):
        batch_size = gen_tok.shape[0]
        cols = gen_tok.shape[1]

        if step + batch_size < len(dataset):
            gen_toks_concat[step: step + batch_size, :cols] = gen_tok
        else:
            gen_toks_concat[step:, :cols] = gen_tok[: len(dataset) - step]
        step += batch_size

    return gen_toks_concat


def get_gold_answers(example):
    """
    helper function from snippet for retrieving correct answers in a SQuAD-like setup
    """
    gold_answers = [answer["text"] for answer in example.answers if answer["text"]]
    if not gold_answers:
        gold_answers = [""]
    return gold_answers


def load_json(json_file_path):
    with open(json_file_path, "r") as file:
        json_data = json.load(file)
    return json_data

def save_json(json_file_path, json_data):
    if not os.path.exists(os.path.dirname(json_file_path)): 
        os.makedirs(os.path.dirname(json_file_path)) 
    
    with open(json_file_path, "w") as output_file:
        json.dump(json_data, output_file, indent=4, sort_keys=True)
    print(json_file_path)


def save_prediction_with_classified_label(total_qid_to_classification_pred, dataset_name, stepNum_result_file, dataName_to_multi_one_zero_file, output_path):
    qid_to_classification_pred = {}
    qid_to_classification_pred_option = {}
    total_stepNum = 0

    for qid in total_qid_to_classification_pred.keys():
        
        if dataset_name != total_qid_to_classification_pred[qid]['dataset_name']:
            continue

        predicted_option = total_qid_to_classification_pred[qid]['prediction']
        
        if predicted_option == 'C':
            total_dict_qid_to_stepNum = load_json(stepNum_result_file)
            stepNum = total_dict_qid_to_stepNum[qid]

        elif predicted_option == 'B':
            stepNum = 1

        elif predicted_option == 'A':
            stepNum = 0

        pred = load_json(dataName_to_multi_one_zero_file[dataset_name][predicted_option])[qid]
        qid_to_classification_pred[qid] = pred
        qid_to_classification_pred_option[qid] = {'prediction' : pred, 'option' : predicted_option, 'stepNum' : stepNum}
        total_stepNum = total_stepNum + stepNum
    
    print('==============')
    save_json(os.path.join(output_path, dataset_name , dataset_name+'.json'), qid_to_classification_pred)
    save_json(os.path.join(output_path, dataset_name, dataset_name+'_option.json'), qid_to_classification_pred_option)
    print('StepNum')
    print(dataset_name + ': ' +str(total_stepNum))

def label_complexity(orig_file_path, zero_file_path, one_file_path, multi_file_path, dataset_name):
    lst_dict_final = []
    with jsonlines.open(orig_file_path, 'r') as input_file:
        for line in input_file:
            dict_question_complexity = {}
            dict_question_complexity['id'] = line['question_id']
            dict_question_complexity['question'] = line['question_text']

            dict_zero = load_json(zero_file_path)
            dict_one = load_json(one_file_path)
            dict_multi = load_json(multi_file_path)

            lst_multi_qid = [i for i in dict_multi.keys()]
            lst_one_qid = [i for i in dict_one.keys()]
            lst_zero_qid = [i for i in dict_zero.keys()]

            if line['question_id'] not in lst_multi_qid + lst_one_qid + lst_zero_qid:
                continue

            dict_question_complexity['dataset_name'] = dataset_name

            lst_total_answer = []

            if line['question_id'] in lst_multi_qid:
                dict_question_complexity['answer'] = 'C' #'multiple'
                lst_total_answer.append('multiple')
            if line['question_id'] in lst_one_qid:
                dict_question_complexity['answer'] = 'B' # 'one'
                lst_total_answer.append('one')
            if line['question_id'] in lst_zero_qid:
                dict_question_complexity['answer'] = 'A' #'zero'
                lst_total_answer.append('zero')
            
            dict_question_complexity['total_answer'] = lst_total_answer

            lst_dict_final.append(dict_question_complexity)

    return lst_dict_final
def count_stepNum(pred_file):
    dict_qid_to_stepNum = {}
    total_stepNum = 0
    stepNum = 0
    new_qid_flag = False

    with open(pred_file, "r") as f:
        for line in f:
            if line == '\n':
                new_qid_flag = True
                if 'qid' in locals():
                    dict_qid_to_stepNum[qid] = stepNum + 1
                    total_stepNum = total_stepNum + stepNum + 1
                stepNum = 0
                continue

            if new_qid_flag:
                qid = line.strip()
                new_qid_flag = False

            if 'Exit? No.' in line:
                stepNum = stepNum + 1
    
    # last qid
    dict_qid_to_stepNum[qid] = stepNum + 1
    total_stepNum = total_stepNum + stepNum + 1

    output_file = '/'.join(pred_file.split('/')[:-1]) + '/stepNum.json'
    save_json(output_file, dict_qid_to_stepNum)

    print(total_stepNum)

    return dict_qid_to_stepNum
