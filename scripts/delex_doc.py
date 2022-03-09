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
    parser = argparse.ArgumentParser(description="delex a file")
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
    parser.add_argument(
        "--dry_run",
        "-d",
        action="store_true",
        help="dry run",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.entity_types == "all":
        ENTITY_TYPES = ALL_TYPES
    else:
        ENTITY_TYPES = [_e.upper() for _e in args.entity_types.split(",")]
    assert len(set(ENTITY_TYPES).intersection(set(ALL_TYPES))) == len(ENTITY_TYPES)

    if args.dry_run:
        sent = ' Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven " . '
        _line, cur_delexed, cur_total = delex_line(
            line=sent, entity_types=ENTITY_TYPES, return_stat=True, dep_types=["subj", "obj", "root"]
        )
        print(_line)
    else:
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

        delexed_portion = round(delexed / total * 100, 1)
        print(f"{delexed_portion}% are delexed")

        SAVE_DIR = os.path.join(
            CWD,
            "data",
            f"{args.save_folder_name}-{delexed_portion}",
        )
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        SAVE_FILE = os.path.join(SAVE_DIR, "train.txt")
        with open(SAVE_FILE, "w") as f:
            f.writelines(delexed_lines)
        print(SAVE_FILE)
