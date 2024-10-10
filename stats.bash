# Helper script to run the stats.py script
# Adjust the path to the results directory
RESULTS_DIR=./results
python3 stats.py -o $RESULTS_DIR --models_subdir statsmodels $RESULTS_DIR
