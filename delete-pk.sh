#!/bin/bash

set -ex

echo "Delete /data/nn_training_data.csv"

rm data/nn_training_data.csv

cp data/nn_training_data_pk_hosszu.csv data/nn_training_data.csv

echo "Done"
