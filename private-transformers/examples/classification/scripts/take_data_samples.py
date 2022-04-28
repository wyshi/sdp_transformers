"""
python classification/scripts/take_data_samples.py -d classification/data/normalized_mask/MNLI/MNLI-entity_only_high-8.63 -n 301
python classification/scripts/take_data_samples.py -d classification/data/original/MNLI -n 301
"""
import argparse
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description="take samples")
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default=None,
        help="dir to the original files",
    )
    parser.add_argument(
        "--sample_n",
        "-n",
        type=int,
        default=None,
        help="sample n",
    )
    args = parser.parse_args()

    return args


def take_samples(data, n):
    np.random.seed(1111)
    sampled_idx = np.random.choice(len(data), size=n, replace=False)
    import pdb

    pdb.set_trace()
    sampled_data = [data[idx] for idx in sampled_idx]

    return sampled_data


def main(args):
    train_file_dir = os.path.join(args.dir, "train.tsv")

    with open(train_file_dir, "r", encoding="utf-8-sig") as fh:
        lines = fh.readlines()

    sampled_lines = take_samples(lines, args.sample_n)

    if "original" not in args.dir:
        for _task in ["MNLI", "QQP", "SST-2", "QNLI"]:
            if _task in args.dir:
                task_name = _task
        save_dir = os.path.join(
            os.path.dirname(args.dir),
            f"{task_name}-sample{args.sample_n}",
            f"{args.dir.rstrip('/').split('/')[-1]}-sample{args.sample_n}",
        )
        # save_dir = f"{args.dir.rstrip('/')}-sample{args.sample_n}"
    else:
        task_name = args.dir.rstrip("/").split("/")[-1]
        save_dir = f"{args.dir.rstrip('/').replace('original', 'normalized_mask')}/{task_name}-sample{args.sample_n}/original-sample{args.sample_n}"
    import pdb

    pdb.set_trace()
    os.makedirs(save_dir, exist_ok=True)

    with open(
        os.path.join(save_dir, "train.tsv"),
        "w",
        encoding="utf-8-sig",
    ) as fh:
        fh.writelines(sampled_lines)

    print(os.path.join(save_dir, "train.tsv"))

    if "mnli" not in save_dir.lower():
        dev_file_dir = os.path.join(args.dir, "dev.tsv")
        test_file_dir = os.path.join(args.dir, "test.tsv")
        # dev set shouldn't be normalized
        os.system(f"cp {dev_file_dir} {save_dir}")
        # test set shouldn't be normalized
        os.system(f"cp {test_file_dir} {save_dir}")
    else:
        dev_file_dir = os.path.join(args.dir, "dev_matched.tsv")
        test_file_dir = os.path.join(args.dir, "test_matched.tsv")
        # dev set shouldn't be normalized
        os.system(f"cp {dev_file_dir} {save_dir}")
        # test set shouldn't be normalized
        os.system(f"cp {test_file_dir} {save_dir}")

        dev_file_dir = os.path.join(args.dir, "dev_mismatched.tsv")
        test_file_dir = os.path.join(args.dir, "test_mismatched.tsv")
        # dev set shouldn't be normalized
        os.system(f"cp {dev_file_dir} {save_dir}")
        # test set shouldn't be normalized
        os.system(f"cp {test_file_dir} {save_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
