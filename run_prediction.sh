#!/bin/bash

# Uncomment and set your OpenAI API key if required for the 'gpt' model
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# Define the lists of configurations
SYSTEMS=("ircot_qa" "oner_qa" "nor_qa")  # Systems: multi, single, zero
MODELS=("gpt4o" "flan-t5-xl" "flan-t5-xxl" "gpt4o_mini")
DATASETS=("nq" "squad" "trivia" "2wikimultihopqa" "hotpotqa" "musique")  # Datasets
LLM_PORT_NUM="8002"  # Port number for the LLM

# Loop through all combinations of SYSTEM, MODEL, and DATASET
for SYSTEM in "${SYSTEMS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
      echo "Running with SYSTEM=$SYSTEM, MODEL=$MODEL, DATASET=$DATASET, LLM_PORT_NUM=$LLM_PORT_NUM"
      bash run_retrieval_train.sh $SYSTEM $MODEL $DATASET $LLM_PORT_NUM
      
      # Check if the last command succeeded
      if [ $? -ne 0 ]; then
        echo "Error running SYSTEM=$SYSTEM, MODEL=$MODEL, DATASET=$DATASET. Skipping..."
        continue
      fi
    done
  done
done


# Loop through all combinations of SYSTEM, MODEL, and DATASET
for SYSTEM in "${SYSTEMS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
      echo "Running with SYSTEM=$SYSTEM, MODEL=$MODEL, DATASET=$DATASET, LLM_PORT_NUM=$LLM_PORT_NUM"
      bash run_retrieval_valid.sh $SYSTEM $MODEL $DATASET $LLM_PORT_NUM
      
      # Check if the last command succeeded
      if [ $? -ne 0 ]; then
        echo "Error running SYSTEM=$SYSTEM, MODEL=$MODEL, DATASET=$DATASET. Skipping..."
        continue
      fi
    done
  done
done



for SYSTEM in "${SYSTEMS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
      echo "Running with SYSTEM=$SYSTEM, MODEL=$MODEL, DATASET=$DATASET, LLM_PORT_NUM=$LLM_PORT_NUM"
      bash run_retrieval_test.sh $SYSTEM $MODEL $DATASET $LLM_PORT_NUM
      
      # Check if the last command succeeded
      if [ $? -ne 0 ]; then
        echo "Error running SYSTEM=$SYSTEM, MODEL=$MODEL, DATASET=$DATASET. Skipping..."
        continue
      fi
    done
  done
done
