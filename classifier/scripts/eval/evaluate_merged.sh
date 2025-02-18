GPU=0
dataname=valid

for model in flan_t5_xl gpt4o gpt4o_mini flan_t5_xxl
do 
  for alpha in 0 0.2 0.4 0.6 0.8 1
  do
    result_path=./outputs/merged_${model}_${alpha}/epoch_20/${dataname}
    model_path=./outputs/merged_${model}_${alpha}/epoch_20/
    
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.evaluate \
      --model_name_or_path ${model_path} \
      --validation_file ./data/${dataname}.json \
      --output_dir ${result_path} \
      --max_seq_length 384 \
      --per_device_eval_batch_size 100 \
      --question_column question \
      --answer_column answer \
      --doc_stride 128 
  done
done


















