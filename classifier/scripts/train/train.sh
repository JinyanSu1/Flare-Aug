    CUDA_VISIBLE_DEVICES=0,1 python -m src.train \
  --model_name_or_path t5-large \
  --train_file ./data/binary/train.json \
  --output_dir outputs/binary \
  --learning_rate 3e-5 \
  --max_seq_length 384 \
  --max_answer_length 30 \
  --num_train_epochs 21 \
  --checkpointing_num 20 \
  --per_device_train_batch_size 64


for model in gpt4o_mini gpt4o flan_t5_xl flan_t5_xxl
do
    CUDA_VISIBLE_DEVICES=0,1 python -m src.train \
  --model_name_or_path t5-large \
  --train_file ./data/${model}/train.json \
  --output_dir outputs/${model} \
  --learning_rate 3e-5 \
  --max_seq_length 384 \
  --max_answer_length 30 \
  --num_train_epochs 21 \
  --checkpointing_num 20 \
  --per_device_train_batch_size 64
done 

