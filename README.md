# Code for What Makes a Language Easy to Deep-learn?

## Set up

1. Set up a virtual environment (e.g., via conda) with a recent python version (we used Python 3.9.5)
1. Within the virtual environment, [install PyTorch](https://pytorch.org/get-started/locally/) according to your OS, GPU availability, and Python package manager.
2. Within the virtual environment, install all other requirements via `pip install -r requirements.txt` 

## Fetch data from experiments with human participants

The data can be obtained via [OSF](https://osf.io/d5ty7/) and should be placed in the `./data` subfolder.
In particular, you need all `LearningExp_*_log.txt` files and the `input_languages.csv` file.

## Main entry point

The main entry point is `train.py`.
Information on command line arguments can be obtained via `python3 train.py -h`.

An exemplary command to run an experiment is

```bash
    python3 train.py --as_humans /data/path/to/experiment.log --seed 1234 --outdir "results-v1"
```

## Scripts to reproduce experiments

Use the following command to reproduce the main experiments from the paper, sweeping over all experiment log files ten times with different random seeds.

```bash
    bash sweep_as_humans.bash
```

Results will be stored in a subfolder `results-v1`.

## Experiments with GPT-3

The main file for running our experiments with GPT-3 is `lang2prompt.py`. It expects `data` directory to be present and filled and will write its outputs to `gpt3-completions`.
You need to specify a language id (S1,B1,S2,...,S5,B5) as a command line argument.

An example call to run the memorization test and the generalization test on language B4 would be:

```bash
    python3 lang2prompt.py B4 --gpt3-mem-test --gpt3-reg-test
```

**Important**: You need to make sure that the shell environment variable `OPENAI_API_KEY` holds your API key and edit the line starting with `openai.organization` with your corresponding organization id.


```bash
    python3 lang2prompt.py B4 --gpt3-mem-test --gpt3-reg-test
```


# Run statistics

Use the following command to reproduce the statistical analysis.

```
    python3 stats.py -o stats-output results-v1
```







