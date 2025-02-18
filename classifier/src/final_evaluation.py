#!/usr/bin/env python3

"""
Combine the classification postprocessing (first code) and the evaluation (second code)
into a single script. You can simply run this one file to perform both steps in sequence.
Make sure you have the same directory structure and the required modules/libraries,
including `postprocess_utils` and the `metrics` folder, available.
"""

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import json
import jsonlines
import re
import uuid
import subprocess
import string
from typing import Dict, Any

# --------------------------------------------------------------------------
#  If you have the "postprocess_utils.py" locally, ensure it's importable:
from utils.data_utils import (
    load_json,
    save_prediction_with_classified_label
)


import _jsonnet
from lib import (
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    get_config_file_path_from_name_or_path,
)

from metrics.drop_answer_em_f1 import DropAnswerEmAndF1
from metrics.support_em_f1 import SupportEmF1Metric
from metrics.answer_support_recall import AnswerSupportRecallMetric
from metrics.squad_answer_em_f1 import SquadAnswerEmF1Metric
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run classification post-processing followed by evaluation in one go."
    )
    parser.add_argument(
        "--model_name",
        type=str,
    )
    parser.add_argument("--classification_result_file", type=str, required=True, help="Path to classification result file.")
    parser.add_argument("--StepNum_result_file", type=str, required=True, help="Path to StepNum result file.")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--retrieval_result_file_dir", type=str, required=True)
    parser.add_argument("--official_evaluation_path", type=str, default =None)
    parser.add_argument("--raw_data_path", type=str, default =None)
    parser.add_argument('--evaluate_type', type=str, default='valid')

    args = parser.parse_args()

    # -------------------------
    # Part 1: Classification Post-Processing
    # -------------------------

    # NOTE: Adjust these paths to match your actual file system.


    stepNum_result_file = args.StepNum_result_file
    classification_result_file = args.classification_result_file
    output_path = args.output_path
    gt_path = args.gt_path
    bm25_retrieval_count_params = {'gpt': {'oner': 6, 'ircot': 3},
                                      'flan': {'oner': 15, 'ircot': 6},
    }
    if 'gpt' in args.model_name:
        one_retrieval_count = bm25_retrieval_count_params['gpt']['oner']
        multi_retrieval_count = bm25_retrieval_count_params['gpt']['ircot']
    elif 'flan' in args.model_name:
        one_retrieval_count = bm25_retrieval_count_params['flan']['oner']
        multi_retrieval_count = bm25_retrieval_count_params['flan']['ircot']
    res_name = args.evaluate_type

    nq_multi_file = os.path.join(args.retrieval_result_file_dir, f'ircot_qa_{args.model_name}_nq____prompt_set_1___bm25_retrieval_count__{multi_retrieval_count}___distractor_count__1', f'prediction__nq_to_nq__{res_name}_subsampled.json') 
    trivia_multi_file = os.path.join(args.retrieval_result_file_dir, f'ircot_qa_{args.model_name}_trivia____prompt_set_1___bm25_retrieval_count__{multi_retrieval_count}___distractor_count__1', f'prediction__trivia_to_trivia__{res_name}_subsampled.json')
    squad_multi_file = os.path.join(args.retrieval_result_file_dir, f'ircot_qa_{args.model_name}_squad____prompt_set_1___bm25_retrieval_count__{multi_retrieval_count}___distractor_count__1', f'prediction__squad_to_squad__{res_name}_subsampled.json')
    musique_multi_file = os.path.join(args.retrieval_result_file_dir, f'ircot_qa_{args.model_name}_musique____prompt_set_1___bm25_retrieval_count__{multi_retrieval_count}___distractor_count__1', f'prediction__musique_to_musique__{res_name}_subsampled.json')
    hotpotqa_multi_file = os.path.join(args.retrieval_result_file_dir, f'ircot_qa_{args.model_name}_hotpotqa____prompt_set_1___bm25_retrieval_count__{multi_retrieval_count}___distractor_count__1', f'prediction__hotpotqa_to_hotpotqa__{res_name}_subsampled.json')
    wikimultihopqa_multi_file = os.path.join(args.retrieval_result_file_dir, f'ircot_qa_{args.model_name}_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__{multi_retrieval_count}___distractor_count__1', f'prediction__2wikimultihopqa_to_2wikimultihopqa__{res_name}_subsampled.json')

    nq_one_file = os.path.join(args.retrieval_result_file_dir, f'oner_qa_{args.model_name}_nq____prompt_set_1___bm25_retrieval_count__{one_retrieval_count}___distractor_count__1', f'prediction__nq_to_nq__{res_name}_subsampled.json') 
    trivia_one_file = os.path.join(args.retrieval_result_file_dir, f'oner_qa_{args.model_name}_trivia____prompt_set_1___bm25_retrieval_count__{one_retrieval_count}___distractor_count__1', f'prediction__trivia_to_trivia__{res_name}_subsampled.json')
    squad_one_file = os.path.join(args.retrieval_result_file_dir, f'oner_qa_{args.model_name}_squad____prompt_set_1___bm25_retrieval_count__{one_retrieval_count}___distractor_count__1', f'prediction__squad_to_squad__{res_name}_subsampled.json')
    musique_one_file = os.path.join(args.retrieval_result_file_dir, f'oner_qa_{args.model_name}_musique____prompt_set_1___bm25_retrieval_count__{one_retrieval_count}___distractor_count__1', f'prediction__musique_to_musique__{res_name}_subsampled.json')
    hotpotqa_one_file = os.path.join(args.retrieval_result_file_dir, f'oner_qa_{args.model_name}_hotpotqa____prompt_set_1___bm25_retrieval_count__{one_retrieval_count}___distractor_count__1', f'prediction__hotpotqa_to_hotpotqa__{res_name}_subsampled.json')
    wikimultihopqa_one_file = os.path.join(args.retrieval_result_file_dir, f'oner_qa_{args.model_name}_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__{one_retrieval_count}___distractor_count__1', f'prediction__2wikimultihopqa_to_2wikimultihopqa__{res_name}_subsampled.json')

    nq_zero_file = os.path.join(args.retrieval_result_file_dir, f'nor_qa_{args.model_name}_nq____prompt_set_1', f'prediction__nq_to_nq__{res_name}_subsampled.json') 
    trivia_zero_file = os.path.join(args.retrieval_result_file_dir, f'nor_qa_{args.model_name}_trivia____prompt_set_1', f'prediction__trivia_to_trivia__{res_name}_subsampled.json')
    squad_zero_file = os.path.join(args.retrieval_result_file_dir, f'nor_qa_{args.model_name}_squad____prompt_set_1', f'prediction__squad_to_squad__{res_name}_subsampled.json')
    musique_zero_file = os.path.join(args.retrieval_result_file_dir, f'nor_qa_{args.model_name}_musique____prompt_set_1', f'prediction__musique_to_musique__{res_name}_subsampled.json')
    hotpotqa_zero_file = os.path.join(args.retrieval_result_file_dir, f'nor_qa_{args.model_name}_hotpotqa____prompt_set_1', f'prediction__hotpotqa_to_hotpotqa__{res_name}_subsampled.json')
    wikimultihopqa_zero_file = os.path.join(args.retrieval_result_file_dir, f'nor_qa_{args.model_name}_2wikimultihopqa____prompt_set_1', f'prediction__2wikimultihopqa_to_2wikimultihopqa__{res_name}_subsampled.json')




    dataName_to_multi_one_zero_file = {
        'musique': {
            'C': musique_multi_file,
            'B': musique_one_file,
            'A': musique_zero_file, 
            
        },
        'hotpotqa': {
            'C': hotpotqa_multi_file,
            'B': hotpotqa_one_file,
            'A': hotpotqa_zero_file,
            
        },
        '2wikimultihopqa': {
            'C': wikimultihopqa_multi_file,
            'B': wikimultihopqa_one_file,
            'A': wikimultihopqa_zero_file,
            
        },
        'nq': {
            'C': nq_multi_file,
            'B': nq_one_file,
            'A': nq_zero_file,
            
        },
        'trivia': {
            'C': trivia_multi_file,
            'B': trivia_one_file,
            'A': trivia_zero_file,
            
        },
        'squad': {
            'C': squad_multi_file,
            'B': squad_one_file,
            'A': squad_zero_file,
            
        },
    }

    # Load classification predictions
    total_qid_to_classification_pred = load_json(classification_result_file)

    # Save new predictions with classification-based step labels
    for data_name in dataName_to_multi_one_zero_file.keys():
        save_prediction_with_classified_label(
            total_qid_to_classification_pred,
            data_name,
            stepNum_result_file,
            dataName_to_multi_one_zero_file,
            output_path
        )

    print("Classification post-processing completed.")
    print("--------------------------------------------------")

    # -------------------------
    # Part 2: Evaluation
    # -------------------------


    def normalize_answer(s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def answer_extractor(potentially_cot: str) -> str:
        """
        Attempts to extract a final short answer from a chain-of-thought string.
        """
        if potentially_cot.startswith('"') and potentially_cot.endswith('"'):
            potentially_cot = potentially_cot[1:-1]

        cot_regex = re.compile(r".* answer is:? (.*)\.?")
        match = cot_regex.match(potentially_cot)
        if match:
            output = match.group(1)
            if output.endswith("."):
                output = output[:-1]
        else:
            output = potentially_cot

        return output

    def load_ground_truths(ground_truth_file_path: str) -> Dict[str, Any]:
        id_to_ground_truths = {}
        with jsonlines.open(ground_truth_file_path, 'r') as input_file:
            for line in input_file:
                qid = line['question_id']
                # Each line['answers_objects'][0]['spans'] is a list of strings
                answer = line['answers_objects'][0]['spans']
                id_to_ground_truths[qid] = answer
        return id_to_ground_truths

    def load_predictions(prediction_file_path: str) -> Dict[str, Any]:
        with open(prediction_file_path, "r") as file:
            id_to_predictions = json.load(file)
        return id_to_predictions

    def save_results(results_dict: Dict[str, Any], output_path: str):
        print(f"Saving eval results to: {output_path}")
        with open(output_path, "w") as file:
            json.dump(results_dict, file, indent=4)

    def calculate_acc(prediction: str, ground_truth: list) -> int:
        for gt in ground_truth:
            if gt in prediction:
                return 1
        return 0

    def evaluate_by_dicts(data_name: str):
        metrics = [SquadAnswerEmF1Metric()]
        pred_json_path = os.path.join(output_path, data_name, f"{data_name}.json")
        id_to_predictions = load_predictions(pred_json_path)
        gt_jsonl_path = os.path.join(gt_path, data_name, f"{args.evaluate_type}_subsampled.jsonl")
        id_to_ground_truths = load_ground_truths(gt_jsonl_path)

        total_acc = 0
        for qid in set(id_to_predictions.keys()):
            ground_truth = id_to_ground_truths[qid]
            prediction = id_to_predictions[qid]

            # Ensure prediction is always a list of strings
            if isinstance(prediction, str):
                # Could be a single string or a bracketed string representation
                if prediction.strip().startswith("[") or prediction.strip().endswith("]"):
                    # e.g. "[answer1, answer2]"
                    pred_list = (
                        prediction.replace('"', "")
                        .replace("[", "")
                        .replace("]", "")
                        .split(",")
                    )
                    prediction = [p.strip() for p in pred_list]
                else:
                    prediction = [prediction]
            elif isinstance(prediction, (list, tuple)):
                prediction = [str(p) for p in prediction]
            else:
                raise ValueError(f"Unexpected prediction type: {type(prediction)}")

            # Extract final answers if there's a chain-of-thought
            prediction = [answer_extractor(p) for p in prediction]

            # Only measure accuracy by the first item (like single-answer)
            # but you could adjust this logic if you want multiple answers
            norm_pred = normalize_answer(prediction[0])
            norm_gts = [normalize_answer(g) for g in ground_truth]
            acc = calculate_acc(norm_pred, norm_gts)
            total_acc += acc

            # Also update the EM/F1 metrics
            metrics[0](prediction, ground_truth)

        total_acc /= len(id_to_predictions)
        evaluation_results = metrics[0].get_metric()
        evaluation_results['acc'] = total_acc

        save_results(
            evaluation_results,
            os.path.join(output_path, data_name, "eval_metic_result_acc.json")
        )

    def official_evaluate_by_dicts(data_name: str):
        """
        Runs official scripts for musique, hotpotqa, 2wikimultihopqa
        which require additional processing and invocation of separate scripts.
        """
        pred_json_path = os.path.join(output_path, data_name, f"{data_name}.json")
        id_to_predictions = load_predictions(pred_json_path)
        gt_jsonl_path = os.path.join(gt_path, data_name, f"{args.evaluate_type}_subsampled.jsonl")
        id_to_ground_truths = load_ground_truths(gt_jsonl_path)

        question_ids = list(id_to_predictions.keys())

        # Convert any list predictions into a single string
        for qid, prediction in list(id_to_predictions.items()):
            if isinstance(prediction, list):
                if len(prediction) == 1:
                    id_to_predictions[qid] = str(prediction[0])
                elif len(prediction) > 1:
                    # Join them with space or any delimiter
                    id_to_predictions[qid] = " ".join(str(e) for e in prediction)
                    print("WARNING: Found a list answer prediction, concatenating it.")

        os.makedirs(".temp", exist_ok=True)

        if data_name == "hotpotqa":
            # Use the official hotpotqa script
            temp_ground_truth_file_path = os.path.join(".temp", uuid.uuid4().hex)
            original_data = read_json(os.path.join(args.raw_data_path, "hotpotqa", "hotpot_dev_distractor_v1.json"))
            filtered_data = [d for d in original_data if d["_id"] in question_ids]
            write_json(filtered_data, temp_ground_truth_file_path)

            temp_prediction_file_path = os.path.join(".temp", uuid.uuid4().hex)
            data = {
                "answer": {qid: str(ans) for qid, ans in id_to_predictions.items()},
                "sp": {qid: [["", 0]] for qid in id_to_predictions.keys()},
            }
            write_json(data, temp_prediction_file_path)

            # We'll cd into official script directory
            # and pass relative paths to that script
            rel_ground_truth = os.path.abspath(temp_ground_truth_file_path)
            rel_prediction = os.path.abspath(temp_prediction_file_path)
            temp_output_file_path = os.path.join(".temp", uuid.uuid4().hex)
            rel_output = os.path.abspath(temp_output_file_path)

            eval_dir = os.path.join(args.official_evaluation_path, "hotpotqa")
            command = (
                f"cd {eval_dir} ; "
                f"python hotpot_evaluate_v1.py {rel_prediction} {rel_ground_truth} > {rel_output}"
            )
            status = subprocess.call(command, shell=True)
            if status != 0:
                raise Exception("Running the official hotpotqa evaluation script failed.")

            # parse the output
            if not os.path.exists(temp_output_file_path):
                raise Exception("The official evaluation output file not found.")

            with open(temp_output_file_path, "r") as file:
                metrics_ = eval(file.read().strip())  # hotpot script prints a dict
                metrics = {
                    "f1": round(metrics_["f1"], 5),
                    "em": round(metrics_["em"], 5),
                    "precision": round(metrics_["prec"], 5),
                    "recall": round(metrics_["recall"], 5),
                    "count": len(id_to_predictions),
                    "acc": round(metrics_["acc"], 5),
                }

            # Cleanup
            os.remove(temp_ground_truth_file_path)
            os.remove(temp_prediction_file_path)
            os.remove(temp_output_file_path)

            save_results(
                metrics,
                os.path.join(output_path, data_name, "eval_metic_result_acc.json")
            )

        elif data_name == "2wikimultihopqa":
            # Use the official 2wikimultihop evaluation script
            temp_ground_truth_file_path = os.path.join(".temp", uuid.uuid4().hex)
            original_data = read_json(os.path.join(args.raw_data_path, "2wikimultihopqa", "dev.json"))
            filtered_data = [d for d in original_data if d["_id"] in question_ids]
            write_json(filtered_data, temp_ground_truth_file_path)

            temp_prediction_file_path = os.path.join(".temp", uuid.uuid4().hex)
            data = {
                "answer": {qid: str(ans) for qid, ans in id_to_predictions.items()},
                "sp": {qid: [["", 0]] for qid in id_to_predictions.keys()},
                "evidence": {qid: ["", "", ""] for qid in id_to_predictions.keys()},
            }
            write_json(data, temp_prediction_file_path)

            rel_ground_truth = os.path.abspath(temp_ground_truth_file_path)
            rel_prediction = os.path.abspath(temp_prediction_file_path)
            alias_file_path = os.path.join(args.raw_data_path,"2wikimultihopqa", "id_aliases.json"
            )
            temp_output_file_path = os.path.join(".temp", uuid.uuid4().hex)
            rel_output = os.path.abspath(temp_output_file_path)

            eval_dir = os.path.join(args.official_evaluation_path, "2wikimultihopqa")
            command = (
                f"cd {eval_dir} ; "
                f"python 2wikimultihop_evaluate_v1.1.py {rel_prediction} {rel_ground_truth} {alias_file_path} > {rel_output}"
            )
            subprocess.call(command, shell=True)

            if not os.path.exists(temp_output_file_path):
                raise Exception("The official evaluation output file not found.")

            with open(temp_output_file_path, "r") as file:
                metrics_ = json.loads(file.read().strip())
                metrics = {
                    "f1": round(metrics_["f1"] / 100, 5),
                    "em": round(metrics_["em"] / 100, 5),
                    "precision": round(metrics_["prec"] / 100, 5),
                    "recall": round(metrics_["recall"] / 100, 5),
                    "count": len(id_to_predictions),
                    "acc": round(metrics_["acc"] / 100, 5),
                }

            # Cleanup
            os.remove(temp_ground_truth_file_path)
            os.remove(temp_prediction_file_path)
            os.remove(temp_output_file_path)

            save_results(
                metrics,
                os.path.join(output_path, data_name, "eval_metic_result_acc.json")
            )

        elif data_name == "musique":
            # Official musique evaluation
            temp_ground_truth_file_path = os.path.join(".temp", uuid.uuid4().hex)
            original_data = read_jsonl(os.path.join(args.raw_data_path, "musique", "musique_ans_v1.0_dev.jsonl"))
            original_keyed = {d["id"]: d for d in original_data}
            filtered_data = [original_keyed[qid] for qid in question_ids]
            write_jsonl(filtered_data, temp_ground_truth_file_path)

            temp_prediction_file_path = os.path.join(".temp", uuid.uuid4().hex)
            data = []
            for qid in question_ids:
                data.append(
                    {
                        "id": qid,
                        "predicted_answer": str(id_to_predictions[qid]),
                        "predicted_support_idxs": [0, 1],
                        "predicted_answerable": True,
                    }
                )
            write_jsonl(data, temp_prediction_file_path)

            rel_ground_truth = os.path.abspath(temp_ground_truth_file_path)
            rel_prediction = os.path.abspath(temp_prediction_file_path)
            temp_output_file_path = os.path.join(".temp", uuid.uuid4().hex)
            rel_output = os.path.abspath(temp_output_file_path)

            eval_dir = os.path.join(args.official_evaluation_path, "musique")
            command = (
                f"cd {eval_dir} ; "
                f"python evaluate_v1.0.py {rel_prediction} {rel_ground_truth} "
                f"--output_filepath {rel_output}"
            )
            subprocess.call(command, shell=True)

            if not os.path.exists(temp_output_file_path):
                raise Exception("The official evaluation output file not found.")

            with open(temp_output_file_path, "r") as file:
                metrics_ = json.loads(file.read().strip())
                metrics = {
                    "f1": round(metrics_["answer_f1"], 3),
                    "em": round(metrics_.get("answer_em", 0.0), 3),
                    "count": len(id_to_predictions),
                    "acc": round(metrics_["answer_acc"], 3),
                }

            # Cleanup
            os.remove(temp_ground_truth_file_path)
            os.remove(temp_prediction_file_path)
            os.remove(temp_output_file_path)

            save_results(
                metrics,
                os.path.join(output_path, data_name, "eval_metic_result_acc.json")
            )

    # Evaluate standard single-hop sets
    for data_name in ['nq', 'trivia', 'squad']:
        evaluate_by_dicts(data_name)

    # Evaluate multi-hop sets with official scripts
    for data_name in ['musique', 'hotpotqa', '2wikimultihopqa']:
        official_evaluate_by_dicts(data_name)

    print("Evaluation completed.")
    print("All done!")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
