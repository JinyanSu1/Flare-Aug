
GPU=0
for model in flan_t5_xl gpt4o gpt4o_mini flan_t5_xxl
do
for alpha in 0.2 0.4 0.6 0.8 1
do
CUDA_VISIBLE_DEVICES=${GPU} python -m src.merge \
   --model_path_1 ./outputs/binary/epoch_20 \
    --model_path_2 ./outputs/${model}/epoch_20 \
    --merged_output_path ./outputs/merged_${model}_${alpha}/epoch_20 \
    --alpha ${alpha}
done
done


