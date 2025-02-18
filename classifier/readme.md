# Preprocess Data
We have already stored processed data in `data` directory, but if you want to do experiments on other models or dataset, preprocess the data in `../predictions`

```
bash scripts/data/preprocess.sh
```

# compute step efficiency:
```
bash scripts/data/step_efficiency.sh
```

# Train cost-optimized classifier and reliability-optimized classifier

```
bash scripts/train/train.sh
```
# Merge the parameters 
```
bash scripts/train/merge.sh
```


- evaluate with classifier
```
bash scripts/eval/evaluate_merged.sh
```
- Get the final RAG evaluation
```
bash scripts/eval/final_evaluation.sh
```









