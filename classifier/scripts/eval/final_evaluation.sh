dataname=valid
GPU=0
BASE_DIR=$(pwd)
for model in flan_t5_xl gpt4o gpt4o_mini flan_t5_xxl
do 
  for alpha in 0 0.2 0.4 0.6 0.8 1
  do
    result_dir=./outputs/merged_${model}_${alpha}/epoch_20/${dataname}/
    classification_result_file=${result_dir}/dict_id_pred_results.json
    output_path=./outputs/merged_${model}_${alpha}/epoch_20/final_evaluation_${dataname}/

    retrieval_result_file_dir=../predictions/${dataname}
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.final_evaluation \
      --model_name ${model} \
      --classification_result_file ${classification_result_file} \
      --StepNum_result_file ${retrieval_result_file_dir}/ircot_qa_${model}/total/stepNum.json \
      --output_path ${output_path} \
      --gt_path ../processed_data \
      --retrieval_result_file_dir ${retrieval_result_file_dir} \
      --official_evaluation_path ../official_evaluation \
      --raw_data_path ${BASE_DIR}/../raw_data \
      --evaluate_type ${dataname}
  done
done


