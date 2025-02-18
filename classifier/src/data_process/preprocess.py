import json
import jsonlines
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))

from ..utils.data_utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="model name.")
parser.add_argument("--output_path", type=str, help="output path.")
parser.add_argument("--original_data_path", type=str, default="processed_data")
parser.add_argument("--data_name", type=str, default="valid")
parser.add_argument("--processed_data_path", type=str, default="predictions")



args = parser.parse_args()
original_data_path = args.original_data_path
data_name = args.data_name
processed_data_path = args.processed_data_path
bm25_retrieval_count_params = {'gpt': {'oner': 6, 'ircot': 3},
                                    'flan': {'oner': 15, 'ircot': 6},
}
if 'gpt' in args.model_name:
    one_retrieval_count = bm25_retrieval_count_params['gpt']['oner']
    multi_retrieval_count = bm25_retrieval_count_params['gpt']['ircot']
elif 'flan' in args.model_name:
    one_retrieval_count = bm25_retrieval_count_params['flan']['oner']
    multi_retrieval_count = bm25_retrieval_count_params['flan']['ircot']

res_name = data_name
    
# Set your file path accordingly
orig_nq_file = os.path.join(original_data_path, "nq", f'{data_name}_subsampled.jsonl')
orig_trivia_file = os.path.join(original_data_path, "trivia", f'{data_name}_subsampled.jsonl')
orig_squad_file = os.path.join(original_data_path, "squad", f'{data_name}_subsampled.jsonl')
orig_musique_file = os.path.join(original_data_path, "musique", f'{data_name}_subsampled.jsonl')
orig_hotpotqa_file = os.path.join(original_data_path, "hotpotqa", f'{data_name}_subsampled.jsonl')
orig_wikimultihopqa_file = os.path.join(original_data_path, "2wikimultihopqa", f'{data_name}_subsampled.jsonl')

nq_multi_file = os.path.join(processed_data_path, data_name, f'ircot_qa_{args.model_name}_nq____prompt_set_1___bm25_retrieval_count__{multi_retrieval_count}___distractor_count__1', f'zero_single_multi_classification__nq_to_nq__{res_name}_subsampled.json') 
trivia_multi_file = os.path.join(processed_data_path, data_name, f'ircot_qa_{args.model_name}_trivia____prompt_set_1___bm25_retrieval_count__{multi_retrieval_count}___distractor_count__1', f'zero_single_multi_classification__trivia_to_trivia__{res_name}_subsampled.json')
squad_multi_file = os.path.join(processed_data_path, data_name, f'ircot_qa_{args.model_name}_squad____prompt_set_1___bm25_retrieval_count__{multi_retrieval_count}___distractor_count__1', f'zero_single_multi_classification__squad_to_squad__{res_name}_subsampled.json')
musique_multi_file = os.path.join(processed_data_path, data_name, f'ircot_qa_{args.model_name}_musique____prompt_set_1___bm25_retrieval_count__{multi_retrieval_count}___distractor_count__1', f'zero_single_multi_classification__musique_to_musique__{res_name}_subsampled.json')
hotpotqa_multi_file = os.path.join(processed_data_path, data_name, f'ircot_qa_{args.model_name}_hotpotqa____prompt_set_1___bm25_retrieval_count__{multi_retrieval_count}___distractor_count__1', f'zero_single_multi_classification__hotpotqa_to_hotpotqa__{res_name}_subsampled.json')
wikimultihopqa_multi_file = os.path.join(processed_data_path, data_name, f'ircot_qa_{args.model_name}_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__{multi_retrieval_count}___distractor_count__1', f'zero_single_multi_classification__2wikimultihopqa_to_2wikimultihopqa__{res_name}_subsampled.json')

nq_one_file = os.path.join(processed_data_path, data_name, f'oner_qa_{args.model_name}_nq____prompt_set_1___bm25_retrieval_count__{one_retrieval_count}___distractor_count__1', f'zero_single_multi_classification__nq_to_nq__{res_name}_subsampled.json') 
trivia_one_file = os.path.join(processed_data_path,data_name, f'oner_qa_{args.model_name}_trivia____prompt_set_1___bm25_retrieval_count__{one_retrieval_count}___distractor_count__1', f'zero_single_multi_classification__trivia_to_trivia__{res_name}_subsampled.json')
squad_one_file = os.path.join(processed_data_path, data_name, f'oner_qa_{args.model_name}_squad____prompt_set_1___bm25_retrieval_count__{one_retrieval_count}___distractor_count__1', f'zero_single_multi_classification__squad_to_squad__{res_name}_subsampled.json')
musique_one_file = os.path.join(processed_data_path, data_name, f'oner_qa_{args.model_name}_musique____prompt_set_1___bm25_retrieval_count__{one_retrieval_count}___distractor_count__1', f'zero_single_multi_classification__musique_to_musique__{res_name}_subsampled.json')
hotpotqa_one_file = os.path.join(processed_data_path, data_name, f'oner_qa_{args.model_name}_hotpotqa____prompt_set_1___bm25_retrieval_count__{one_retrieval_count}___distractor_count__1', f'zero_single_multi_classification__hotpotqa_to_hotpotqa__{res_name}_subsampled.json')
wikimultihopqa_one_file = os.path.join(processed_data_path, data_name, f'oner_qa_{args.model_name}_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__{one_retrieval_count}___distractor_count__1', f'zero_single_multi_classification__2wikimultihopqa_to_2wikimultihopqa__{res_name}_subsampled.json')

nq_zero_file = os.path.join(processed_data_path, data_name, f'nor_qa_{args.model_name}_nq____prompt_set_1', f'zero_single_multi_classification__nq_to_nq__{res_name}_subsampled.json') 
trivia_zero_file = os.path.join(processed_data_path, data_name, f'nor_qa_{args.model_name}_trivia____prompt_set_1', f'zero_single_multi_classification__trivia_to_trivia__{res_name}_subsampled.json')
squad_zero_file = os.path.join(processed_data_path, data_name, f'nor_qa_{args.model_name}_squad____prompt_set_1', f'zero_single_multi_classification__squad_to_squad__{res_name}_subsampled.json')
musique_zero_file = os.path.join(processed_data_path, data_name, f'nor_qa_{args.model_name}_musique____prompt_set_1', f'zero_single_multi_classification__musique_to_musique__{res_name}_subsampled.json')
hotpotqa_zero_file = os.path.join(processed_data_path, data_name, f'nor_qa_{args.model_name}_hotpotqa____prompt_set_1', f'zero_single_multi_classification__hotpotqa_to_hotpotqa__{res_name}_subsampled.json')
wikimultihopqa_zero_file = os.path.join(processed_data_path, data_name, f'nor_qa_{args.model_name}_2wikimultihopqa____prompt_set_1', f'zero_single_multi_classification__2wikimultihopqa_to_2wikimultihopqa__{res_name}_subsampled.json')

output_path = os.path.join(args.output_path, args.model_name)

lst_nq = label_complexity(orig_nq_file, nq_zero_file, nq_one_file, nq_multi_file, 'nq')
lst_trivia = label_complexity(orig_trivia_file, trivia_zero_file, trivia_one_file, trivia_multi_file, 'trivia')
lst_squad = label_complexity(orig_squad_file, squad_zero_file, squad_one_file, squad_multi_file, 'squad')
lst_musique = label_complexity(orig_musique_file, musique_zero_file, musique_one_file, musique_multi_file, 'musique')
lst_hotpotqa = label_complexity(orig_hotpotqa_file, hotpotqa_zero_file, hotpotqa_one_file, hotpotqa_multi_file, 'hotpotqa')
lst_wikimultihopqa = label_complexity(orig_wikimultihopqa_file, wikimultihopqa_zero_file, wikimultihopqa_one_file, wikimultihopqa_multi_file, '2wikimultihopqa')


lst_total_data = lst_musique + lst_hotpotqa + lst_wikimultihopqa + lst_nq + lst_trivia + lst_squad

save_json(os.path.join(output_path, f'{data_name}.json'), lst_total_data)
