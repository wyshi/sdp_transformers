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
    sampled_data = [data[idx] for idx in sampled_idx]

    return sampled_data


def main(args):
    train_file_dir = os.path.join(args.dir, "train.tsv")
    dev_file_dir = os.path.join(args.dir, "dev.tsv")
    test_file_dir = os.path.join(args.dir, "test.tsv")
    with open(train_file_dir, "r", encoding="utf-8-sig") as fh:
        lines = fh.readlines()

    sampled_lines = take_samples(lines, args.sample_n)

    if "original" not in args.dir:
        save_dir = f"{args.dir.rstrip('/')}-sample{args.sample_n}"
    else:
        task_name = args.dir.rstrip("/").split("/")[-1]
        save_dir = (
            f"{args.dir.rstrip('/').replace('original', 'normalized_mask')}/{task_name}/original-sample{args.sample_n}"
        )
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
    # dev set shouldn't be normalized
    os.system(f"cp {dev_file_dir} {save_dir}")
    # test set shouldn't be normalized
    os.system(f"cp {test_file_dir} {save_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
