for data_name in train valid test
do
  for model in gpt4o flan_t5_xl flan_t5_xxl gpt4o_mini
  do
    python -m src.data_process.preprocess \
      --model_name $model \
      --output_path data \
      --original_data_path ../processed_data \
      --data_name $data_name \
      --processed_data_path ../predictions
  done
done



