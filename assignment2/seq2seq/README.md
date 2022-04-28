## Assignment : Part 2 :Sequence-to-Sequence Model

# Seq2Seq for Machine Translation


## 0. Requirements
#### General
- Python (developed on 3.7)
- unzip

#### Python Packages
- tensorflow (deep learning library)
- tensorboard (Deep Learning Graphs and Charts)
- matplotlib (for visaulization)
- sklearn (Data Split)
- pandas (Read Json)


## 1. Pre-processing
- First, prepare data. Donwload Spider data and GloVe and nltk corpus
(downloaded to `$HOME/data`)

- Second, load the data and select the Train and Test data, update the paths in data.py and run:
```
python -m data.py
Use data_collection_explained.ipynb to understand the Data collection Process and Graphs
```
- Third, clean, standardize and vectorize the data:
```
Use data_preparation_explained to understand the Data Preparation Process
```
## 2. Training and Inference
- First Experiment : NMT with Attention Layer (Custom Embedding)
```
Run nmt_attention_model.ipynb to train NMT with Attention Layer  Model (Custom Embedding)
```
See train.py for model architecture 


- Second Experiment : NMT with Attention Layer (Glove Embedding)
```
Run nmt_attention_with_embedding.ipynb to train NMT with Attention Layer  Model (Glove Embedding)
```
See train.py for model architecture

- Third Experiment : Encoder Decoder Architechture 
```
Run enc_dec.ipynb to train and evaluate basic Encoder Decoder Model
```

## 3. Accuracy

1. Below accuracy is Exact Match accuarcy , i.e. when the Target SQL and Actual SQL match excped case conversion and whitespace
2. A seperate Testing data is utilzed for Test accuracy

|      Model                                    | Exact Match (%) |
| ----------------------------------------------|:---------------:|
| Encoder Decoder                               | 2.80  
| NMT with Attention Layer (Custom Embedding)   | 45.xx
| NMT with Attention Layer (Glove Embedding)    | 29.xx

