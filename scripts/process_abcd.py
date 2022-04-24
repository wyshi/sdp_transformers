import os, sys
import json
import re
import argparse
from tqdm import tqdm
import numpy as np
from nltk.tokenize import sent_tokenize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import convert_abcd_line, MAP, EOS, decide_delex_level
from policy_functions import delex_line

unique_sents = set()

CWD = os.getcwd()

FILE = os.path.join(CWD, "../abcd/data/abcd_v1.1.json")
SAVE_DIR = os.path.join(CWD, ".")

with open(
    FILE,
) as fh:
    data = json.load(fh)

train_data = data["train"]
valid_data = data["dev"]
test_data = data["test"]


DELEX_TERMS = [
    "<account_id>",
    "<amount>",
    "<email>",
    "<name>",
    "<order_id>",
    "<phone>",
    "<pin_number>",
    "<street_address>",
    "<username>",
    "<zip_code>",
    "<password>",
]


def build_non_enumerable(train_data):
    non_enumerable = {"personal": [], "order": [], "product": []}
    for dial in train_data:
        for key in non_enumerable:
            non_enumerable[key].extend(list(dial["scenario"][key].keys()))
    for key in non_enumerable:
        non_enumerable[key] = list(set(non_enumerable[key]))
    non_enumerable["order"].remove("num_products")
    non_enumerable["order"].remove("packaging")
    non_enumerable["order"].remove("payment_method")
    non_enumerable["order"].remove("state")
    non_enumerable["product"].remove("amounts")
    return non_enumerable


NON_ENUMERABLE = build_non_enumerable(train_data)

def dedup(text):
    sents = sent_tokenize(text)
    if len(sents) > 0:
        if sents[-1] == '!':
            sents.pop(-1)
            sents[-1] = sents[-1] + '!'
        elif sents[-1] == '?':
            sents.pop(-1)
            sents[-1] = sents[-1] + '?'
    
    for i, sent in enumerate(sents):
        if sent in unique_sents:
            sents[i] = '<SENT_MASK>'
        else:
            unique_sents.add(sent)
    
    text = ' '.join(sents)
    return text


def delexicalization(args, scene, conversation, use_single_mask=True):
    """Given all the utterances within a converation and the scenario, delexicalize the
    non enumerable entities. Inputs:
        - scene: a dict with detail, personal_info and order info
        - conversation: a list of tuples where each tuple is (speaker, text, action, pred)
    Returns:
        - delex: a list pf tuples where the text has been delexicalized
    """
    delexed = []
    ENTITY_TYPES, DEP_TYPES, POS_TYPES, PREDICTOR = decide_delex_level(args.contextual_level)
    for speaker, text in conversation:
        if args.deduplicate:
            text = dedup(text)
        # if "it's 48281 " in text:
        #     import pdb

        #     pdb.set_trace()
        # must be in this order to prevent clash
        text, cur_delexed, cur_total = delex_line(
            line=text,
            entity_types=ENTITY_TYPES,
            return_stat=True,
            dep_types=DEP_TYPES,
            predictor=PREDICTOR,
            pos_types=POS_TYPES,
            use_single_mask_token=True,
            concat_consecutive_special_tokens=True,
        )
        for slot, slotval in scene["personal"].items():
            if slot in NON_ENUMERABLE["personal"] and str(slotval).lower() in text.lower():
                mask_token = "<MASK>" if use_single_mask else f"<{slot}>"
                text = re.sub(re.escape(str(slotval)), mask_token, text, flags=re.IGNORECASE)
        for slot, slotval in scene["order"].items():
            if slot in NON_ENUMERABLE["order"] and str(slotval).lower() in text.lower():
                mask_token = "<MASK>" if use_single_mask else f"<{slot}>"
                text = re.sub(re.escape(str(slotval)), mask_token, text, flags=re.IGNORECASE)
        # product amount might clobber phone or account_id
        for slots, slotvals in scene["product"].items():
            slot = slots[:-1]  # drop the 's'
            for slotval in slotvals:
                if slot in NON_ENUMERABLE["product"] and str(slotval).lower() in text.lower():
                    mask_token = "<MASK>" if use_single_mask else f"<{slot}>"
                    text = re.sub(re.escape(str(slotval)), mask_token, text, flags=re.IGNORECASE)

        delexed.append([speaker, text])

    # check for password
    pswd_re = re.compile(
        "(?=[a-zA-Z]*[0-9])(?=[0-9]*[a-zA-Z])[a-zA-Z0-9]{10,11}",
        flags=re.IGNORECASE,
    )
    mask_token = "<MASK>" if use_single_mask else f"<password>"
    for i, (speaker, text) in enumerate(delexed):
        if "A password has been generated" in text:
            try:
                delexed[i + 1][1] = re.sub(
                    pswd_re,
                    mask_token,
                    delexed[i + 1][1],
                )
                delexed[i + 2][1] = re.sub(
                    pswd_re,
                    mask_token,
                    delexed[i + 2][1],
                )
            except IndexError:
                print("something wrong")
    return delexed


def save_to_file_for_classification(split, records, delexed_records):
    # original data
    save_dir = os.path.join(SAVE_DIR, "abcd_classification_original")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"{split}.json")
    with open(save_file, "w") as fh:
        json.dump(records, fh)
    print(save_file)

    # delexed data
    save_dir = os.path.join(SAVE_DIR, "abcd_classification_delex")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"{split}.json")
    with open(save_file, "w") as fh:
        if split == "train":
            json.dump(delexed_records, fh)
        else:
            json.dump(records, fh)
    print(save_file)


def save_to_file(file_name, split, lines, delexed_lines):
    # original data
    save_dir = os.path.join(SAVE_DIR, file_name.replace("delex", "original"))
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"{split}.txt")
    with open(save_file, "w") as fh:
        fh.writelines(lines)
    print(save_file)

    # delexed data
    save_dir = os.path.join(SAVE_DIR, file_name)
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"{split}.txt")
    with open(save_file, "w") as fh:
        if split == "train":
            fh.writelines(delexed_lines)
        else:
            fh.writelines(lines)
    print(save_file)


def convert_data(
    args,
    data,
    split,
):
    lines = []
    delexed_lines = []
    total_cnt = 0
    delexed_cnt = 0

    if split == "train" and args.sample_n is not None:
        data = take_samples(data, args.sample_n)

    for dial in data:
        for utt in dial["original"]:
            sent = convert_abcd_line(utt[0], utt[1])
            # sent = MAP[utt[0]] + utt[1] + "\n\n"
            lines.append(sent)
        lines.append(EOS)

        for utt in dial["delexed"]:
            sent = convert_abcd_line(utt["speaker"], utt["text"])
            # sent = MAP[utt["speaker"]] + utt["text"] + "\n\n"
            delexed_lines.append(sent)
            total_cnt += len(utt["text"].split())
            sensitive_tokens = [tok for tok in utt["text"].split() if tok in DELEX_TERMS]
            delexed_cnt += len(sensitive_tokens)

        delexed_lines.append(EOS)

    print(f"{split}: portion, {round(delexed_cnt/total_cnt*100, 5)}")

    return lines, delexed_lines


def convert_and_delex_data(
    args,
    data,
    split,
):
    lines = []
    delexed_lines = []
    total_cnt = 0
    delexed_cnt = 0
    delex_portion = 0
    if split == "train" and args.sample_n is not None:
        data = take_samples(data, args.sample_n)

    for dial in tqdm(data):
        for utt in dial["original"]:
            sent = convert_abcd_line(utt[0], utt[1])
            # sent = MAP[utt[0]] + utt[1] + "\n\n"
            lines.append(sent)
        lines.append(EOS)

        if split == "train":
            _delex_line = delexicalization(args, dial["scenario"], dial["original"], use_single_mask=True)
            for utt in _delex_line:
                sent = convert_abcd_line(utt[0], utt[1])
                total_cnt += len(sent.split())
                delexed_cnt += len([tok for tok in sent.split() if tok in DELEX_TERMS or tok == "<MASK>"])
                delexed_lines.append(sent)
            delexed_lines.append(EOS)

    if split == "train":
        delex_portion = round(delexed_cnt / total_cnt * 100, 5)
        print(f"{split}: portion, {delex_portion}")

    return lines, delexed_lines, delex_portion


def convert_data_for_classification(
    data,
    split,
):
    records = []
    delexed_records = []
    total_cnt = 0
    delexed_cnt = 0
    for dial in data:
        label = dial["scenario"]["subflow"]
        lines = []
        for utt in dial["original"]:
            sent = convert_abcd_line(utt[0], utt[1])
            # sent = MAP[utt[0]] + utt[1] + "\n\n"
            lines.append(sent)
        records.append(["".join(lines), label])

        delexed_lines = []
        for utt in dial["delexed"]:
            sent = convert_abcd_line(utt["speaker"], utt["text"])
            # sent = MAP[utt["speaker"]] + utt["text"] + "\n\n"
            delexed_lines.append(sent)
            total_cnt += len(utt["text"].split())
            sensitive_tokens = [tok for tok in utt["text"].split() if tok in DELEX_TERMS]
            delexed_cnt += len(sensitive_tokens)
        delexed_records.append(["".join(delexed_lines), label])

    print(f"{split}: portion, {round(delexed_cnt/total_cnt*100, 5)}")

    return records, delexed_records


def take_samples(data, n):
    np.random.seed(1111)
    sampled_idx = np.random.choice(len(data), size=n, replace=False)
    sampled_data = [data[idx] for idx in sampled_idx]

    return sampled_data


def main(args):
    for data, split in zip([train_data, valid_data, test_data], ["train", "valid", "test"]):
        if args.task == "original_generation":
            lines, delexed_lines = convert_data(
                args,
                data,
                split,
            )
            save_folder = f"abcd_delex-sample{args.sample_n}" if args.sample_n else "abcd_delex"
            save_to_file(save_folder, split, lines, delexed_lines)
        elif args.task == "delex_and_generation":
            lines, delexed_lines, delex_portion = convert_and_delex_data(
                args,
                data,
                split,
            )
            if split == "train":
                train_delex_portion = delex_portion
            save_folder = (
                f"abcd_my_delex-{args.contextual_level}_{train_delex_portion}-sample{args.sample_n}"
                if args.sample_n
                else f"abcd_my_delex-{args.contextual_level}_{train_delex_portion}"
            )
            save_to_file(save_folder, split, lines, delexed_lines)
        elif args.task == "classification":
            records, delexed_records = convert_data_for_classification(data, split)
            save_to_file_for_classification(split, records, delexed_records)


def parse_args():
    parser = argparse.ArgumentParser(description="delex a file")
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default=None,
        choices=["original_generation", "delex_and_generation", "classification"],
        help="tasks",
    )
    parser.add_argument(
        "--contextual_level",
        "-cl",
        type=str,
        choices=["entity_only_low", "entity_only_medium", "entity_only_high", "no_pronoun", "default", "root", "SRL"],
        default=None,
        help="contextual level",
    )
    parser.add_argument(
        "--sample_n",
        "-n",
        type=int,
        default=None,
        help="sample n",
    )
    parser.add_argument(
        "--deduplicate",
        "-dedup",
        type=bool,
        default=False,
        help="deduplicate data when doing delexicalization",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
