# Sentiment Analysis Project

This repositopry contains the three models used for a comparative study of three approaches to binary sentiment classification of movie reviews.
The models predict weather a review expresses negative or positive sentiment.

The project compares a Support-Vector Machine with a Naive-Bayes layer, a Long short-term memory recurrent neural network, and a finetuned transformer (pretrained from Huggingface).

- Support Vector Machine with a Naive-Bayes feature weighting layer (NB-SVM)
- Bidirectional Long Short-Term Memory with a self-attetion layer (BiLSTM)
- RoBERTa, finetuned on the IMDB Dataset, a transformer

## Dataset
The labeled part of the Stanford IMDB Dataset for Sentiment Analysis from Huggingface is used with it's predetermined split.    
For more information here: https://huggingface.co/datasets/stanfordnlp/imdb
## Repository structure

```text
.
├── common/
│   └── tokenizer.py              # Shared text tokenizer
├── dataset/
│   ├── train.csv                 # Official IMDb training split
│   └── test.csv                  # Official IMDb test split
├── lstm/
│   ├── utils/                    # Data, vocabulary, model, and training helpers
│   ├── lstm_train.py             # BiLSTM training pipeline
│   ├── lstm_eval.py              # BiLSTM test evaluation
│   └── plot.py                   # BiLSTM training-history plot
├── nb_svm/
│   ├── utils/                    # NB-SVM preprocessing and vocabulary helpers
│   ├── nb_svm_train.py           # NB-SVM training pipeline
│   └── nb_svm_eval.py            # NB-SVM test evaluation
├── roberta/
│   └── roberta.py                # RoBERTa inference and test evaluation
├── tests/                        # Unit tests for tokenizer and vocabulary
├── evaluation.py                 # Final model comparison and plots
├── get_dataset.py                # Short script for downloading the official IMDB Dataset
├── prepare_env.sh                # Shell script to create the virtual environment with requirements
└── requirements.txt              # Project dependencies
```

Generated model files, preprocessed features, prediction CSVs, and plots are
stored inside the corresponding model directories or in the project root.

## Models

### NB-SVM

Linear classifier (linear SVM) with weighted features using Naive Bayes log-count ratios. The vocabulary consists of unigrams and bigrams. This model is based on the approach proposed by Wang and Manning [1].

### BiLSTM

The BiLSTM model with self-attention layer based on the lab excersise with some improvements:
- Improved tokenizer
- Trainable embedding layer initialized with GloVe vectors for tokens for available tokens
- AdamW optimizer with L2 Regulariztation to improve generalizaion ability (with limited success)

### RoBERTa

Pretrained RoBERTa model finetuned on HF IMDB Dataset. Still to be changed


## Experimental results

All final results should be measured on the complete official IMDb test split
of 25,000 reviews.

| Model | Test accuracy | Training Parameters |
| --- | ---: | --- |
| NB-SVM | 90.84% | C=1, $\alpha$=1  |
| BiLSTM | 90.43% | LR=1e-3 (1e-4 for embedding), L2= 1e-4, epochs=15|
| RoBERTa | TODO | None |

Interpretation: The LSTM with self-attention layer fails to surpass the NB-SVM model while requiring considerably more time/resources to train. The transformer-based model surpasses the NB-SVM by a couple of percentage points while much more complex and not viable to train without access to an HPC cluster.In conclusion, a surprisingly good performance by the relatively simple NB-SVM.

## Environment setup

Create the virtual environment and install the dependencies from the project
root:

```bash
bash prepare_env.sh
source .venv/bin/activate
```

## Downloading the dataset

Download the official Hugging Face IMDB train and test splits:

```bash
python get_dataset.py
```

This creates `dataset/train.csv` and `dataset/test.csv`.

## Running the models

Run all commands from  project root with the venv activated.

### NB-SVM



```bash
# Preprocess raw dataset
python -m nb_svm.utils.preprocessing
# Train classifier
python -m nb_svm.nb_svm_train
# Evaluate classifier on the test set and write predictions into a csv
python -m nb_svm.nb_svm_eval
```

### BiLSTM


```bash
# Train BiLSTM
python -m lstm.lstm_train --epochs <number of epochs>
# Evaluate BiLSTM on test set and write predicitons into a csv  
python -m lstm.lstm_eval
# Plot training history
python -m lstm.plot
```

### RoBERTa

```bash
# Load pretrained model, evaluate on test set and write prediction into csv
python -m roberta.roberta
```

## Comparing the models

After all results.csv are created, run the following script to compare the three models:

```bash
python evaluation.py
```

This creates:

- `model_comparison.png`: bar chart of model test accuracies
- `model_confusion_matrices.png`: confusion matrices for all three models

## Unit tests

In the development of the BiLSTM and NB SVM, unit tests for the vocab and the tokenizer were created to ensure requirements are met.
To run them, run the follownig commands:

```bash
# Tokenizer test
python -m pytest tests/tokenizer_test.py -v
# Vocabulary test
python -m pytest tests/vocab_test.py -v

```

## References
[1] S. Wang and C. D. Manning, “Baselines and Bigrams: Simple, Good Sentiment and Topic Classification,” Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics, 2012. [Paper](https://aclanthology.org/P12-2018/)

## Collaborators

| Name | Matrikelnummer | Degree program |
| --- | --- | --- |
| Fabio von Daak | 03787787 | Elektrotechnik und Informationstechnik |
| Simon Matthäus Blasbichler | 03790798 | Elektrotechnik und Informationstechnik |
| Daniel Demianiw | 03783597 | Elektrotechnik und Informationstechnik |
