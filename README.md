# Fast or Better? Balancing Accuracy and Cost in Retrieval-Augmented Generation with Flexible User Control



## Installation
Create a conda environment as follows:
```bash
$ conda create -n Flare-Aug python=3.8
$ conda activate Flare-Aug
$ pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
$ pip install -r requirements.txt
```

## Prepare Retriever Server
After installing the conda environment, you should setup the retriever server as follows:
```bash
$ wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
$ wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
$ shasum -a 512 -c elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
$ tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz
$ cd elasticsearch-7.10.2/
$ ./bin/elasticsearch # start the server
# pkill -f elasticsearch # to stop the server
```

Start the elasticsearch server on port 9200 (default), and then start the retriever server as shown below.
```bash
uvicorn serve:app --port 8000 --app-dir retriever_server
```


## Datasets
* You can download raw data
```bash
$ bash ./download/raw_data.sh

# Build index
python retriever_server/build_index.py {dataset_name} # hotpotqa, 2wikimultihopqa, musique
```

```bash
# Download Natural Question
$ mkdir -p raw_data/nq
$ cd raw_data/nq
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
$ gzip -d biencoder-nq-dev.json.gz
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
$ gzip -d biencoder-nq-train.json.gz

# Download TriviaQA
$ cd ..
$ mkdir -p trivia
$ cd trivia
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz
$ gzip -d biencoder-trivia-dev.json.gz
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz
$ gzip -d biencoder-trivia-train.json.gz

# Download SQuAD
$ cd ..
$ mkdir -p squad
$ cd squad
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz
$ gzip -d biencoder-squad1-dev.json.gz
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz
$ gzip -d biencoder-squad1-train.json.gz

# Download Wiki passages. For the singe-hop datasets, we use the Wikipedia as the document corpus.
$ cd ..
$ mkdir -p wiki
$ cd wiki
$ wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
$ gzip -d psgs_w100.tsv.gz

# Build index 
$ python retriever_server/build_index.py wiki
```

We provide the preprocessed datasets in [google drive](https://drive.google.com/drive/folders/1T7YIKAOIkMsohpoVODZdYl7pCGYJV-h_)

## Prepare LLM Server
After indexing for retrieval is done, you can verify the number of indexed documents in each of the four indices by executing the following command in your terminal: `curl localhost:9200/_cat/indices`. You should have 4 indices and expect to see the following sizes: HotpotQA (5,233,329), 2WikiMultihopQA (430,225), MuSiQue (139,416), and Wiki (21,015,324).

Next, if you want to use FLAN-T5 series models, start the llm_server (for flan-t5-xl and xxl) by running:
```bash
MODEL_NAME={model_name} uvicorn serve:app --port 8010 --app-dir llm_server # model_name: flan-t5-xxl, flan-t5-xl
```




## Modify config files
| gen_models   | name         | engine                     |
|-------------|-------------|----------------------------|
| `gpt_chat`  | `gpt4o`     | `gpt-4o-2024-11-20`        |
| `gpt_chat`  | `gpt4o_mini` | `gpt-4o-mini-2024-07-18`   |
| `llm_api`   | `flan_t5_xl`  | `google/flan-t5-xl`       |
| `llm_api`   | `flan_t5_xxl` | `google/flan-t5-xxl`      |




## Run Three Different Retrieval Strategies
```
bash run_prediction.sh
```
We provide all the results for three different retrieval strategies in [google drive](https://drive.google.com/drive/folders/1T7YIKAOIkMsohpoVODZdYl7pCGYJV-h_). 






## Acknowledgement
We refer to the repository of and [Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG) and [IRCoT](https://github.com/StonyBrookNLP/ircot) as a skeleton code.



