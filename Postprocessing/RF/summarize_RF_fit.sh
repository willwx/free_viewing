#!/bin/bash

script="Summarize RF fit"

papermill "${script}.ipynb" "Log - ${script} - Fix on.ipynb"
