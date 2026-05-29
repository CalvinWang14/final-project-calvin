#!/bin/bash
set -e

sudo python3 -m pip install --upgrade pip

sudo python3 -m pip install \
    beautifulsoup4 \
    lxml \
    vaderSentiment \
    nltk \
    scikit-learn \
    umap-learn \
    boto3 \
    pandas \
    numpy

sudo python3 -c "
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
"
