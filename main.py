# include built-in libraries
import argparse

from transformers import AutoTokenizer

import arguments
from src import utils

# include self-defined libraries
from src.mind import download_extract_small_mind, preprocess
from src.trainer import Trainer


def _train(args):
    trainer = Trainer(args)
    trainer.train()


def _eval(args):
    trainer = Trainer(args)
    trainer.eval()


def main():
    parser = argparse.ArgumentParser(
        description="Arguments for NRMS model",
        fromfile_prefix_chars="@",
        allow_abbrev=False,
    )
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args
    subparsers = parser.add_subparsers(
        dest="mode", help="Mode of the process: train or test"
    )

    train_parser = subparsers.add_parser("train", help="Training phase")
    arguments.add_train_arguments(train_parser)
    eval_parser = subparsers.add_parser("eval", help="Evaluation phase")
    arguments.add_eval_arguments(eval_parser)

    args = parser.parse_args()
    if args.mode == "train":
        _train(args)
    elif args.mode == "eval":
        _eval(args)


if __name__ == "__main__":
    # defining the constants values
    PRETRAINED_TOKENIZER = "vinai/phobert-base"

    train_path, validation_path = download_extract_small_mind(
        size="small", dest_path="./data", clean_zip_file=False
    )

    print(
        "Training path: {}\nValidation path: {}".format(
            train_path, validation_path
        )
    )

    Tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TOKENIZER)
    flag = preprocess(train_path=train_path, tokenizer=Tokenizer)
    print(flag)
    main()
