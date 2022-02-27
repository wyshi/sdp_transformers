"""
python scripts/delex_doc.py -o data/wikitext-2-raw/train.txt \
    -s wiki_person \
    -e person
"""
import os, sys
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy_functions import delex_line, ALL_TYPES

CWD = os.getcwd()

def parse_args():
    parser = argparse.ArgumentParser(
        description="delex a file"
    )
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
        "--entity_types",
        "-e",
        type=str,
        help="entity types to protect",
    )
    args = parser.parse_args()


    return args


if __name__ == "__main__":
    args = parse_args()

    ENTITY_TYPES = [_e.upper() for _e in args.entity_types.split(",")]
    assert len(set(ENTITY_TYPES).intersection(set(ALL_TYPES))) == len(ENTITY_TYPES)

    with open(
        args.original_file,
        # "/local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/train.txt",
        "r",
        encoding="utf8",
    ) as f:
        lines = f.readlines()

    delexed_lines = []
    delexed, total = 0, 0
    for line in tqdm(lines):
        _line, cur_delexed, cur_total = delex_line(line=line, entity_types=ENTITY_TYPES, return_stat=True)
        delexed += cur_delexed
        total += cur_total
        delexed_lines.append(_line)

    delexed_portion = round(delexed/total*100, 1)
    print(f"{delexed_portion}% are delexed")
    
    SAVE_DIR = os.path.join(CWD, "data", f"{args.save_folder_name}-{delexed_portion}", )
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    SAVE_FILE = os.path.join(SAVE_DIR,"train.txt")
    with open(SAVE_FILE, "w") as f:
        f.writelines(delexed_lines)
    print(SAVE_FILE)
