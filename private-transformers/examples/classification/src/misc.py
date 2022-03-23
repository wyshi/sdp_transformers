import os, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
)
from utils import SPECIAL_TOKENS_MAP, MASK_TOKEN


def add_special_tokens(
    tokenizer,
    data_args,
):
    if "abcd" in data_args.task_name:
        tokenizer.add_tokens(
            [
                "SYS:",
                "USR:",
                "ACT:",
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
        )
    else:
        tokenizer.add_tokens(MASK_TOKEN, special_tokens=True)
        # tokenizer.add_tokens(list(SPECIAL_TOKENS_MAP.values()), special_tokens=True)
    return tokenizer
