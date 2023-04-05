import argparse
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Raviv1
from learning_experiment import LearningExp
from modeling import MLP, RelaxedLinear, TiedRnnGS
from training import Trainer

try:
    import wandb
except ImportError:
    print("WandB logging not available, pip install wandb?")
    wandb = None


def main():
    """Simulate a Learning experiment with neural network agents"""

    parser = argparse.ArgumentParser()
    parser.add_argument("input_data", help="Path to Experiment Log")
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random init and shuffling"
    )

    ### MODEL ARGS
    parser.add_argument(
        "--rnn_cell",
        choices=["rnn", "lstm", "gru"],
        default="lstm",
        help="RNN cell for both sender and receiver",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=50, help="Number of hidden units per layer"
    )
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers")

    ### TRAINING ARGS
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--epochs_per_block",
        type=int,
        default=1,
        help="How many epoch per blocks, defaults to 1",
    )

    # we use hidden size for all params
    # parser.add_argument(
    #     "--embed_dim", type=int, default=10, help="Dimension of symbol embedding"
    # )

    parser.add_argument("--input2hidden", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--hidden2output", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--con_weight", type=float, default=0.1)

    parser.add_argument(
        "--as_humans",
        action="store_true",
        default=False,
        help="Train exactly as humans in What makes a language easy to learn",
    )

    ### WandB related arguments
    parser.add_argument("--project", default="easy-to-learn")
    parser.add_argument("--notes", help="Notes for weights and biases")
    parser.add_argument("--tags", nargs="+", help="Tags for weights and biases")


    ### Tied weights option
    parser.add_argument("--tied_weights",
            default="all",
            choices=["all", "within", "between", "none"],
            help="Parameter sharing strategy"
    )

    parser.add_argument(
        "--outdir", help="Store outputs in this dir (path relative to cwd)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Do not track on weights and biases",
    )

    args = parser.parse_args()

    # Load data
    learningexp = LearningExp.load(args.input_data)


    ### FIX ALL SEEDS for reproducibility
    effective_seed = args.seed + int(learningexp.info["Participant ID"])
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)

    ### Build model
    input_dim = Raviv1.num_shapes + 2  # OH-enc of shapes + 2 for sin/cos of radial angle
    hidden_size = args.hidden_size
    embed_dim = args.hidden_size
    vocab_size = len(Raviv1.alphabet) + 1  # + 1 for EOS token
    print("VOCAB SIZE (incl EOS):", vocab_size)
    max_length = Raviv1.max_length
    print("MAX LENGTH (excl EOS):", max_length)
    if args.input2hidden == "linear":
        input2hidden = RelaxedLinear(input_dim, hidden_size)
    else:
        input2hidden = MLP(input_dim, hidden_size, hidden_size)

    if args.hidden2output == "linear":
        hidden2output = RelaxedLinear(hidden_size, hidden_size)
    else:
        hidden2output = MLP(hidden_size, hidden_size, hidden_size)

    print("Building model with tied_weights={args.tied_weights}")
    model = TiedRnnGS(
        input2hidden,
        hidden2output,
        vocab_size,
        embed_dim,
        hidden_size,
        max_length,
        1.0,  # temperature -- doesn't matter as we don't train via GS
        cell=args.rnn_cell,
        trainable_temperature=False,
        straight_through=False,
        tied_weights=args.tied_weights
    )
    print(model)
    if torch.cuda.is_available():
        model.cuda()


    wb_config = vars(args)
    lang = learningexp.lang
    wb_config["Language/InputCondition"] = lang.name
    wb_config["Language/StructureScore"] = lang.get_unique_attribute("StructureScore")
    wb_config["Language/StructureBin"] = lang.get_unique_attribute("StructureBin")
    wb_config["Language/GroupSize"] = lang.get_unique_attribute("GroupSize")
    print(wb_config)

    if not args.debug and wandb is not None:
        wandb.init(
            project=args.project, config=wb_config, notes=args.notes, tags=args.tags
        )
        wandb.watch(model)

    # Prep output
    os.makedirs(args.outdir, exist_ok=True)
    time_str = datetime.datetime.now().strftime('%y%m%d')
    lang_str = learningexp.info['Language']
    filename = f"LearningExp_{time_str}_{lang_str}_{effective_seed:05d}_log.txt"
    outfile = os.path.join(args.outdir, filename)

    logging_freq = min(args.epochs_per_block, 5)
    trainer = Trainer(
        Raviv1,
        model,
        learningexp,
        lr=args.lr,
        batch_size=args.batch_size,
        logging_freq=logging_freq,
        con_weight=args.con_weight,
        use_wandb=not args.debug,
        outfile=outfile,
        seed=effective_seed
    )

    if args.as_humans:
        trainer.train_exactly_as_humans(
            num_iterations=args.iterations, epochs_per_block=args.epochs_per_block
        )
    else:
        trainer.train(
            num_iterations=args.iterations, epochs_per_block=args.epochs_per_block
        )


if __name__ == "__main__":
    main()
