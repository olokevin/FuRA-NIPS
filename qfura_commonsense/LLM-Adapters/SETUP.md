# LLM-Adapters dataset

The commonsense reproduction uses the LLM-Adapters benchmark and its training
mixture. We do **not** vendor the data files (the training mixture alone is
~115 MB and the eight evaluation tasks add ~115 MB).

## Download

```bash
cd commonsense/LLM-Adapters

# Training set (commonsense_170k.json)
mkdir -p ft-training_set
curl -L -o ft-training_set/commonsense_170k.json \
    https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/ft-training_set/commonsense_170k.json

# Evaluation datasets (boolq / piqa / social_i_qa / hellaswag / winogrande / ARC / openbookqa)
git clone --depth 1 https://github.com/AGI-Edgerunners/LLM-Adapters.git _tmp_llm_adapters
cp -r _tmp_llm_adapters/dataset .
rm -rf _tmp_llm_adapters
```

After running this, the layout should be:

```
commonsense/LLM-Adapters/
  ft-training_set/commonsense_170k.json
  dataset/{boolq,piqa,social_i_qa,hellaswag,winogrande,ARC-Challenge,ARC-Easy,openbookqa}/test.json
  commonsense_evaluate.py
  evaluate.py
  ...
```

The training scripts read `${DATA_DIR}/ft-training_set/commonsense_170k.json`
(default `DATA_DIR=LLM-Adapters`) and the eval scripts read
`${DATA_DIR}/<task>/test.json` (default `DATA_DIR=commonsense/LLM-Adapters/dataset`).
