"""
python /local/data/wyshi/sdp_transformers/scripts/take_sample_wiki.py -d /local/data/wyshi/sdp_transformers/data/wiki_entity_all_mask_consec-16.4 -n 10
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
    sampled_data = [data[idx] for idx in sampled_idx]

    return sampled_data


def main(args):
    train_file_dir = os.path.join(args.dir, "train.txt")

    with open(
        train_file_dir,
        "r",
    ) as fh:
        lines = fh.readlines()

    title_ids = [i for i, line in enumerate(lines) if line.startswith(" =")]
    sampled_title_ids = take_samples(title_ids, args.sample_n)

    # with open(
    #     "/local/data/wyshi/sdp_transformers/data/wikitext-2-raw/train.txt",
    #     "r",
    # ) as fh:
    #     lines2 = fh.readlines()

    # title_ids2 = [i for i, line in enumerate(lines2) if line.startswith(" =")]
    import pdb

    pdb.set_trace()
    sampled_lines = []
    for title_id in sampled_title_ids:
        stop_id = (
            title_ids[title_ids.index(title_id) + 1] if title_ids.index(title_id) + 1 < len(title_ids) else len(lines)
        )
        sampled_lines.extend(lines[title_id:stop_id])

    save_dir = os.path.join(
        os.path.dirname(args.dir),
        f"sample{args.sample_n}",
        f"{args.dir.rstrip('/').split('/')[-1]}-sample{args.sample_n}",
    )

    import pdb

    pdb.set_trace()
    os.makedirs(save_dir, exist_ok=True)

    with open(
        os.path.join(save_dir, "train.txt"),
        "w",
    ) as fh:
        fh.writelines(sampled_lines)

    print(os.path.join(save_dir, "train.txt"))

    dev_file_dir = os.path.join(args.dir, "valid.txt")
    test_file_dir = os.path.join(args.dir, "test.txt")
    # dev set shouldn't be normalized
    os.system(f"cp {dev_file_dir} {save_dir}")
    # test set shouldn't be normalized
    os.system(f"cp {test_file_dir} {save_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
