# Paraphrase Detection using Siamese Neural Networks

This document describes the development of a system to identify paraphrased content within textual data.

## Problem Statement

Accurately detecting paraphrased sentences amidst large amounts of text is a challenge. Paraphrased sentences convey the same meaning but use different wording, making traditional text-matching techniques ineffective. Sophisticated algorithms are needed to discern semantic similarity between sentences.

## Motivation

Plagiarism is a serious concern in academia, compromising the integrity of institutions and the value of scholarly work. Detecting paraphrased content can discourage plagiarism and promote academic honesty.

## Approach

### 1. Data and Preprocessing

We initially used the MSR Paraphrase Dataset containing over 5,000 entries [MSR dataset](https://www.kaggle.com/datasets/doctri/microsoft-research-paraphrase-corpus?select=msr_paraphrase_train.txt)
. Preprocessing steps included stop word removal and sentence lemmatization.

### 2. Word Embedding and Siamese Network

We employed Word2Vec for sentence embedding and trained a Siamese Neural Network. This network has three LSTM layers for each branch (sentence and paraphrase), totaling six LSTM layers. The specific details are as follows:

#### Sentence Branch:

* LSTM Layer 1: 128 units
* LSTM Layer 2: 64 units
* LSTM Layer 3: 32 units

#### Paraphrase Branch:

* LSTM Layer 1: 128 units
* LSTM Layer 2: 64 units
* LSTM Layer 3: 32 units

![Siamese Neural Network](#)

The contrastive loss function was used for training:

![Contrastive Loss](#)

This model achieved an accuracy of 35%.

### 3. Improvements for Better Accuracy

To improve accuracy, we made several changes:

* **Dataset:** We switched to the PAWS dataset available on Hugging Face [PAWS dataset](https://huggingface.co/datasets/paws).

* **Embedding Technique:** We replaced Word2Vec with Sentence BERT (specifically, the `paraphrase-multilingual-mpnet-base-v2` model) for word embedding. This significantly improved accuracy to 55%.

![Updated Code](#)

Experimenting with a different Sentence BERT model (`all-MiniLM-L12-v2`) yielded similar accuracy.

These adjustments demonstrate the importance of dataset and embedding techniques in Siamese Neural Network performance.

## Conclusion

We achieved an overall accuracy of 55% over four months. While this is a substantial improvement from the initial 35%, there's room for further enhancement.

Systematically refining our approach through datasets and embedding techniques highlights the importance of data quality and preprocessing methods in machine learning models. The shift from MSR Paraphrase Dataset to PAWS and from Word2Vec to Sentence BERT was crucial for improved accuracy.

Despite these advancements, the model's performance can be further optimized. Future work could explore:

* More advanced neural network architectures
* Hyperparameter fine-tuning
* Integration of additional linguistic features

Our experience underscores the iterative nature of machine learning projects, where continuous evaluation and adaptation are key to achieving optimal results.
