import os, sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import convert_abcd_line, MAP, EOS

CWD = os.getcwd()

FILE = os.path.join(CWD, "abcd/data/abcd_v1.1.json")
SAVE_DIR = os.path.join(CWD, "data/abcd")

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
]


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


def save_to_file(split, lines, delexed_lines):
    # original data
    save_dir = os.path.join(SAVE_DIR, "abcd_original")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"{split}.txt")
    with open(save_file, "w") as fh:
        fh.writelines(lines)
    print(save_file)

    # delexed data
    save_dir = os.path.join(SAVE_DIR, "abcd_delex")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"{split}.txt")
    with open(save_file, "w") as fh:
        if split == "train":
            fh.writelines(delexed_lines)
        else:
            fh.writelines(lines)
    print(save_file)


def convert_data(
    data,
    split,
):
    lines = []
    delexed_lines = []
    total_cnt = 0
    delexed_cnt = 0
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


for data, split in zip([train_data, valid_data, test_data], ["train", "valid", "test"]):
    lines, delexed_lines = convert_data(
        data,
        split,
    )
    records, delexed_records = convert_data_for_classification(data, split)
    save_to_file_for_classification(split, records, delexed_records)
    # save_to_file(split, lines, delexed_lines)
