import argparse

from utils.tools import get_configs_of
from preprocessor import ravdess


def main(config):
    if "RAVDESS" in config["dataset"]:
        ravdess.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    config, *_ = get_configs_of(args.dataset)
    main(config)
