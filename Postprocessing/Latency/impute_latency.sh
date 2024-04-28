#!/bin/bash

script="Impute latency"
papermill -p hier False "${script}.ipynb" "Log - ${script}.ipynb"
papermill -p hier True "${script}.ipynb" "Log - ${script} - Hier.ipynb"