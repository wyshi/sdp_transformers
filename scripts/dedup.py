import hashlib
import spacy
import argparse

nlp = spacy.load("en_core_web_sm")


def parse_args():
    parser = argparse.ArgumentParser(description="dedup a file")
    parser.add_argument(
        "--save_folder_name",
        "-s",
        type=str,
        help="the folder name to save the normalized txt",
    )
    parser.add_argument(
        "--original_file",
        "-o",
        type=str,
        help="the original file name to delex",
    )
    parser.add_argument(
        "--dry_run",
        "-d",
        action="store_true",
        help="dry run",
    )
    args = parser.parse_args()

    return args
