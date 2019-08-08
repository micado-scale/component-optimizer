#!/bin/bash

set -ex

echo "Delete /data/nn_training_data.csv"

rm data/nn_training_data.csv

cp data/grafana_for_mta_cloud_optimizer.csv data/nn_training_data.csv

echo "Done"
