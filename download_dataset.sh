#!/usr/bin/env bash
set -e

DATA_DIR="data"
DATASET="lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"

mkdir -p "$DATA_DIR"

kaggle datasets download \
    -d "$DATASET" \
    -p "$DATA_DIR" \
    --unzip

# Rename to something easier to work with
mv "$DATA_DIR/IMDB Dataset.csv" "$DATA_DIR/imdb_reviews.csv"

echo "Dataset downloaded to $DATA_DIR/imdb_reviews.csv"