import os, sys
import csv

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
)
from policy_functions import delex_line
from utils import NORMALIZE_MAP, decide_delex_level

from transformers.data.processors.glue import (
    glue_processors,
    MnliProcessor,
    Sst2Processor,
    QqpProcessor,
    QnliProcessor,
)

from tqdm import tqdm
import pdb
from glob import glob
import argparse


def _write_tsv(output_file, rows, quotechar=None):
    """Reads a tab separated value file."""
    with open(output_file, "w", encoding="utf-8-sig") as f:
        print(output_file)
        csvwriter = csv.writer(f, delimiter="\t", quotechar=quotechar)
        csvwriter.writerows(rows)


class NormalizedMnliProcessor(MnliProcessor):
    def normalize_and_write_examples(self, input_dir, output_dir, contextual_level):
        train_file_dir = os.path.join(input_dir, "train.tsv")
        dev_file_dir = os.path.join(input_dir, "dev_matched.tsv")
        test_file_dir = os.path.join(input_dir, "test_matched.tsv")

        examples, delexed_portion = self._normalize_examples(
            lines=self._read_tsv(train_file_dir),
            set_type="train",
            contextual_level=contextual_level,
        )
        output_dir = f"{output_dir}-{delexed_portion}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"train.tsv")
        _write_tsv(output_file, rows=examples)

        # dev set shouldn't be normalized
        os.system(f"cp {dev_file_dir} {output_dir}")
        # test set shouldn't be normalized
        os.system(f"cp {test_file_dir} {output_dir}")

    def _normalize_examples(self, lines, set_type, contextual_level):
        """Creates examples for the training, dev and test sets."""
        examples = []
        delexed, total = 0, 0
        ENTITY_TYPES, DEP_TYPES, POS_TYPES, PREDICTOR = decide_delex_level(contextual_level)
        for (i, line) in enumerate(tqdm(lines, desc="normalize")):
            if i == 0:
                examples.append(line)
                continue
            # guid = f"{set_type}-{i}"
            line[8], cur_delexed, cur_total = delex_line(
                line=line[8],
                entity_types=ENTITY_TYPES,
                return_stat=True,
                dep_types=DEP_TYPES,
                predictor=PREDICTOR,
                pos_types=POS_TYPES,
            )
            delexed += cur_delexed
            total += cur_total
            line[9], cur_delexed, cur_total = delex_line(
                line=line[9],
                entity_types=ENTITY_TYPES,
                return_stat=True,
                dep_types=DEP_TYPES,
                predictor=PREDICTOR,
                pos_types=POS_TYPES,
            )
            # label = None if set_type == "test" else line[1]
            delexed += cur_delexed
            total += cur_total
            examples.append(line)
        delexed_portion = round(delexed / total * 100, 2)
        return examples, delexed_portion


class NormalizedSst2Processor(Sst2Processor):
    def normalize_and_write_examples(self, input_dir, output_dir, contextual_level):
        train_file_dir = os.path.join(input_dir, "train.tsv")
        dev_file_dir = os.path.join(input_dir, "dev.tsv")
        test_file_dir = os.path.join(input_dir, "test.tsv")

        examples, delexed_portion = self._normalize_examples(
            lines=self._read_tsv(train_file_dir),
            set_type="train",
            contextual_level=contextual_level,
        )
        output_dir = f"{output_dir}-{delexed_portion}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"train.tsv")
        _write_tsv(output_file, rows=examples)

        # dev set shouldn't be normalized
        os.system(f"cp {dev_file_dir} {output_dir}")
        # test set shouldn't be normalized
        os.system(f"cp {test_file_dir} {output_dir}")

    def _normalize_examples(self, lines, set_type, contextual_level):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1 if set_type == "test" else 0
        delexed, total = 0, 0
        ENTITY_TYPES, DEP_TYPES, POS_TYPES, PREDICTOR = decide_delex_level(contextual_level)
        for (i, line) in enumerate(tqdm(lines, desc="normalize")):
            if i == 0:
                examples.append(line)
                continue
            # guid = f"{set_type}-{i}"
            line[text_index], cur_delexed, cur_total = delex_line(
                line=line[text_index],
                entity_types=ENTITY_TYPES,
                return_stat=True,
                dep_types=DEP_TYPES,
                predictor=PREDICTOR,
                pos_types=POS_TYPES,
            )
            # label = None if set_type == "test" else line[1]
            delexed += cur_delexed
            total += cur_total
            examples.append(line)
        delexed_portion = round(delexed / total * 100, 2)
        return examples, delexed_portion


class NormalizedQqpProcessor(QqpProcessor):
    def normalize_and_write_examples(self, input_dir, output_dir, contextual_level):
        train_file_dir = os.path.join(input_dir, "train.tsv")
        dev_file_dir = os.path.join(input_dir, "dev.tsv")
        test_file_dir = os.path.join(input_dir, "test.tsv")

        examples, delexed_portion = self._normalize_examples(
            lines=self._read_tsv(train_file_dir),
            set_type="train",
            contextual_level=contextual_level,
        )
        output_dir = f"{output_dir}-{delexed_portion}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"train.tsv")
        _write_tsv(output_file, rows=examples)

        # dev set shouldn't be normalized
        os.system(f"cp {dev_file_dir} {output_dir}")
        # test set shouldn't be normalized
        os.system(f"cp {test_file_dir} {output_dir}")

    def _normalize_examples(self, lines, set_type, contextual_level):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        q1_index = 1 if test_mode else 3
        q2_index = 2 if test_mode else 4
        examples = []
        delexed, total = 0, 0
        ENTITY_TYPES, DEP_TYPES, POS_TYPES, PREDICTOR = decide_delex_level(contextual_level)
        for (i, line) in enumerate(tqdm(lines, desc="normalize")):
            if i == 0:
                examples.append(line)
                continue
            # guid = f"{set_type}-{i}"

            try:
                line[q1_index], cur_delexed, cur_total = delex_line(
                    line=line[q1_index],
                    entity_types=ENTITY_TYPES,
                    return_stat=True,
                    dep_types=DEP_TYPES,
                    predictor=PREDICTOR,
                    pos_types=POS_TYPES,
                )
                # label = None if set_type == "test" else line[1]
                delexed += cur_delexed
                total += cur_total

                line[q2_index], cur_delexed, cur_total = delex_line(
                    line=line[q2_index],
                    entity_types=ENTITY_TYPES,
                    return_stat=True,
                    dep_types=DEP_TYPES,
                    predictor=PREDICTOR,
                    pos_types=POS_TYPES,
                )
                # label = None if set_type == "test" else line[1]
                delexed += cur_delexed
                total += cur_total

                examples.append(line)
            except IndexError:
                continue

        delexed_portion = round(delexed / total * 100, 2)
        return examples, delexed_portion


class NormalizedQnliProcessor(QnliProcessor):
    def normalize_and_write_examples(self, input_dir, output_dir, contextual_level):
        train_file_dir = os.path.join(input_dir, "train.tsv")
        dev_file_dir = os.path.join(input_dir, "dev.tsv")
        test_file_dir = os.path.join(input_dir, "test.tsv")

        examples, delexed_portion = self._normalize_examples(
            lines=self._read_tsv(train_file_dir),
            set_type="train",
            contextual_level=contextual_level,
        )
        output_dir = f"{output_dir}-{delexed_portion}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"train.tsv")
        _write_tsv(output_file, rows=examples)

        # dev set shouldn't be normalized
        os.system(f"cp {dev_file_dir} {output_dir}")
        # test set shouldn't be normalized
        os.system(f"cp {test_file_dir} {output_dir}")

    def _normalize_examples(self, lines, set_type, contextual_level):
        """Creates examples for the training, dev and test sets."""
        examples = []
        delexed, total = 0, 0
        ENTITY_TYPES, DEP_TYPES, POS_TYPES, PREDICTOR = decide_delex_level(contextual_level)
        for (i, line) in enumerate(tqdm(lines, desc="normalize")):
            if i == 0:
                examples.append(line)
                continue
            # guid = f"{set_type}-{i}"
            line[1], cur_delexed, cur_total = delex_line(
                line=line[1],
                entity_types=ENTITY_TYPES,
                return_stat=True,
                dep_types=DEP_TYPES,
                predictor=PREDICTOR,
                pos_types=POS_TYPES,
            )
            # label = None if set_type == "test" else line[1]
            delexed += cur_delexed
            total += cur_total

            line[2], cur_delexed, cur_total = delex_line(
                line=line[2],
                entity_types=ENTITY_TYPES,
                return_stat=True,
                dep_types=DEP_TYPES,
                predictor=PREDICTOR,
                pos_types=POS_TYPES,
            )
            # label = None if set_type == "test" else line[1]
            delexed += cur_delexed
            total += cur_total
            examples.append(line)
        delexed_portion = round(delexed / total * 100, 2)
        return examples, delexed_portion


normalized_glue_processors = {
    "mnli": NormalizedMnliProcessor,
    "sst-2": NormalizedSst2Processor,
    "qqp": NormalizedQqpProcessor,
    "qnli": NormalizedQnliProcessor,
}


def get_output_dir(input_dir, contextual_level):
    output_dir = os.path.join(
        input_dir.replace("original", "normalized_mask"), input_dir.split("/")[-1] + f"-{contextual_level}"
    )
    print(output_dir)
    return output_dir


def main(task, contextual_level):
    DEBUG = False
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/original")
    data_dir_suffix = {
        "sst-2": "GLUE-SST-2",
        "mnli": "MNLI",
        "qqp": "QQP",
        "qnli": "QNLI",
    }[task]

    data_dir = os.path.join(DATA_DIR, data_dir_suffix)
    processor = normalized_glue_processors[task]()
    output_dir = get_output_dir(input_dir=data_dir, contextual_level=contextual_level)
    processor.normalize_and_write_examples(
        input_dir=data_dir, output_dir=output_dir, contextual_level=contextual_level
    )
    if DEBUG:
        original_processor = glue_processors[task]()
        original_exs = original_processor.get_train_examples(data_dir)

        now_dir = [
            _dir
            for _dir in glob(os.path.join("/".join(output_dir.split("/")[:-1]), "*"))
            if _dir.startswith(output_dir)
        ][0]
        now_exs = processor.get_train_examples(now_dir)
        import pdb

        pdb.set_trace()


def parse_args():
    parser = argparse.ArgumentParser(description="delex a file")
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        choices=normalized_glue_processors.keys(),
        help="tasks",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    for contextual_level in tqdm(list(NORMALIZE_MAP.keys())[::-1]):
        main(args.task, contextual_level)
