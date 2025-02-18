for model_name in flan_t5_xl flan_t5_xxl gpt4o gpt4o_mini
do
  for data_name in valid test
  do
    python -m src.data_process.step_efficiency \
      --model_name $model_name \
      --set_type $data_name \
      --retrieval_result_file_dir ../predictions
  done
done

    