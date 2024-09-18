import argparse

from src.models import lstm

def add_program_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model_class",
        type=str,
        required=True,
        action="store",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        action="store",
    )
    parser.add_argument(
        "--density_lambda",
        type=float,
        required=True,
        action="store",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        required=True,
        action="store",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        action="store",
    )
    parser.add_argument(
        "--physics_penalty_lambda",
        type=float,
        default=None,
        action="store",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        required=True,
        action="store",
    )
