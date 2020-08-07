#!/bin/bash

jupyter nbconvert --no-input --execute --to html --ExecutePreprocessor.kernel_name=python3 case_load_vs_case_delta.ipynb
