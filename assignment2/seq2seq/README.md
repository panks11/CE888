## Assignment : Part 2 :Sequence-to-Sequence Model

# Seq2Seq for Machine Translation


## 0. Requirements
#### General
- Python (developed on 3.7)
- unzip

#### Python Packages
- tensorflow (deep learning library)
- tensorboard (Deep Learning Graphs and Charts)
- nltk (NLP tools)
- tqdm (progress bar)
- matplotlib (for visaulization)
- sklearn (Data Split)


## 1. Pre-processing
First, prepare data. Donwload Spider data and GloVe and nltk corpus
(downloaded to `$HOME/data`)
```

Second, load the data and select the Train and Test data:
```
python -m data.py
Use data_collection_explained.ipynb to understand the Data collection Process and Graphs

Third, clean, standardize and vectorize the data:
```
Use data_preparation_explained to understand the Data Preparation Process