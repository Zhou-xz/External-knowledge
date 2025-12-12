## Requirements
    - Python==3.8
    - PyTorch==1.12.0
    - transformers==4.15.0
    - torchmetrics==0.7.0
    - SpaCy==3.3.1
    - scikit-learn==1.0.2
    - numpy==1.21.5
    - pytorch-lightning =1.3.5
    - einops==0.4.0
    - en-core-web-sm==3.3.0
## Files
```
├── code
│   ├── utils
│   │   ├── __init__.py
│   │   ├── aste_datamodule.py
|   |   └── aste_result.py
│   ├── model
│   │   ├── seq2mat.py
│   │   ├── table.py
│   │   ├── boundary_contrastive.py
│   │   ├── char_cnn.py
│   │   ├── cross_attention.py
│   │   ├── fusion_layer.py
│   │   ├── gcn.py
│   │   ├── gcn2.py
│   │   ├── interaction_attention.py
│   │   ├── matching_layer.py
│   │   ├── multihead_attention.py
│   │   ├── table_encoder
│   │   |   └── resnet.py
|   |   └── bdtf_model.py
|   ├── aste_train.py
|   └── bash
│       ├── aste.sh
│       ├── V1
│       │   ├── aste_14lap.sh
│       │   ├── aste_14res.sh
│       │   ├── aste_15res.sh
│       |   └── aste_16res.sh
│       └── V2/...
└── data
    └── V1
    │   ├── 14res
    |   │   ├── 14res_pair
    │   │   │   ├── train_pair.pkl
    │   │   │   ├── dev_pair.pkl
    │   │   |   └── test_pair.pkl
    |   │   ├── train.json
    |   │   ├── dev.json
    |   │   ├── test.json
    |   │   ├── char2idx.json
    |   │   ├── idx2word.json
    |   │   └── word2idx.json
    │   ├── 14lap/...
    │   ├── 15res/...
    |   └── 16res/...
    ├── V2/...
    ├── v1_data_process.py
    ├── v1_word2ids.py
    ├── v2_data_process.py
    └── v2_word2ids.py
```

## Pre-processing steps
  - Download the pretrained glove embeddings and unzip it to ' /embedding/GloVe/'
    (http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip)
  - Download the pretrained "Word Embedding of Amazon Product Review Corpus" and unzip it to './embedding/Particular/' (https://zenodo.org/record/3370051)
  - Move "senticnet_word.txt" to ' /senticNet/'
  - Run V2 or V1 preprocessing files
        V2：
            python ./data/v2_word2ids.py
            python ./data/v2_data_process.py
        V1:
            python ./data/v2_word2ids.py
            python ./data/v2_data_process.py

## Training stage
  - Enter the corresponding folder firstly
        cd './code/bash/V2/' or cd './code/bash/V1/'
  - Run the corresponding .sh file 
        For example:  bash aste_14res.sh
