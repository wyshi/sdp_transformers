"""
python scripts/delex_wiki.py -o data/wikitext-2-raw/train.txt \
    -s wiki_contextual_no_pronoun \
    -e all \
    -cl no_pronoun \
    -d

python scripts/delex_wiki.py -o data/wikitext-2-raw/train.txt \
    -s wiki_contextual_default \
    -e all \
    -cl default

python scripts/delex_wiki.py -o data/wikitext-2-raw/train.txt \
    -s wiki_contextual_root \
    -e all \
    -cl root

python scripts/delex_wiki.py -o data/wikitext-2-raw/train.txt \
    -s wiki_contextual_SRL \
    -e all \
    -cl SRL

# "entity_only", "no_pronoun", "default", "root", "SRL"

# entity_only
python scripts/delex_wiki.py -o data/wikitext-2-raw/train.txt \
    -s wiki_entity_all \
    -e all \
    -cl entity_only

python scripts/delex_wiki.py -o data/wikitext-2-raw/train.txt \
    -s wiki_entity_person_org_date_gpe \
    -e person,org,date,gpe \
    -cl entity_only

python scripts/delex_wiki.py -o data/wikitext-2-raw/train.txt \
    -s wiki_entity_person \
    -e person \
    -cl entity_only
"""
import os, sys
import argparse
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy_functions import delex_line
from utils import ALL_TYPES

CWD = os.getcwd()

SRL_MODEL_PATH = (
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
)


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
    # parser.add_argument(
    #     "--use_SRL",
    #     "-us",
    #     action="store_true",
    #     help="use SRL",
    # )
    parser.add_argument(
        "--contextual_level",
        "-cl",
        type=str,
        choices=["entity_only", "no_pronoun", "default", "root", "SRL"],
        default="default",
        help="entity_only: entities, no_pronoun:entity+subj+obj+PROPN, default: entity + subj, obj, PROPN, PRON; root: additional root; SRL: include predicate from SRL",
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

    if args.contextual_level == "entity_only":
        DEP_TYPES, POS_TYPES, PREDICTOR = None, None, None
    else:
        PREDICTOR = None
        if args.contextual_level == "no_pronoun":
            DEP_TYPES = [
                "subj",
                "obj",
            ]
            POS_TYPES = [
                "PROPN",  # proper noun, Mike
                # "PRON",  # pronoun, He
            ]
        elif args.contextual_level == "default":
            DEP_TYPES = [
                "subj",
                "obj",
            ]
            POS_TYPES = [
                "PROPN",  # proper noun, Mike
                "PRON",  # pronoun, He
            ]
        elif args.contextual_level == "root":
            DEP_TYPES = ["subj", "obj", "root"]
            POS_TYPES = [
                "PROPN",  # proper noun, Mike
                "PRON",  # pronoun, He
            ]
        elif args.contextual_level == "SRL":
            DEP_TYPES = ["subj", "obj", "root"]
            POS_TYPES = ["PROPN", "PRON", "VERB"]  # proper noun, Mike  # pronoun, He

    print(DEP_TYPES)
    print(POS_TYPES)
    print(PREDICTOR)
    print(ENTITY_TYPES)
    if args.dry_run:
        sent = "Did I already tell you I'm getting a divorce?"  # "Did you hear Alice is getting divorced??"  #' Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven " . '
        sent = "What are you going to do about custody of the kids?"  # "Did you hear Alice is getting divorced??"  #' Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven " . '
        # sent = "Did you hear Alice is getting divorced??"  # "Did you hear Alice is getting divorced??"  #' Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven " . '
        sent = ' Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven " . '
        # sent = "weiyan shi's birthday is January 23."
        # sent = "Paul gave his book to Mary"
        # sent = "The rock thrown by Paul broke the window"
        # sent = (
        #     "k-19 exploits our substantial collective fear of nuclear holocaust to generate cheap hollywood tension ."
        # )
        # sent = "a very well-made , funny and entertaining picture ."
        # sent = "i do n't think i laughed out loud once "
        _line, cur_delexed, cur_total = delex_line(
            line=sent,
            entity_types=ENTITY_TYPES,
            return_stat=True,
            dep_types=DEP_TYPES,
            predictor=PREDICTOR,
            pos_types=POS_TYPES,
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
            _line, cur_delexed, cur_total = delex_line(
                line=line,
                return_stat=True,
                entity_types=ENTITY_TYPES,
                dep_types=DEP_TYPES,
                predictor=PREDICTOR,
                pos_types=POS_TYPES,
            )
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
        else:
            input(f"{SAVE_DIR} already exists, do you want to overwrite? press any key to continue")
        SAVE_FILE = os.path.join(SAVE_DIR, "train.txt")
        with open(SAVE_FILE, "w") as f:
            f.writelines(delexed_lines)
        print(SAVE_FILE)
        os.system(f"cp {args.original_file.replace('train', 'valid')} {os.path.join(SAVE_DIR, 'valid.txt')}")
        os.system(f"cp {args.original_file.replace('train', 'test')} {os.path.join(SAVE_DIR, 'test.txt')}")