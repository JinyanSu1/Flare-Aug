# Preprocess Data
We have already stored processed data in `data` directory, but if you want to do experiments on other models or dataset, preprocess the data in `../predictions`

```
bash scripts/data/preprocess.sh
```






(1) compute step efficiency:
```
bash scripts/data/step_efficiency.sh
```
(2) evaluate with classifier
```
bash scripts/eval/evaluate_merged.sh
```
(3) Get the final RAG evaluation
```
bash scripts/eval/final_evaluation.sh
```





# Train the model
```
bash scripts/train/train.sh
```




