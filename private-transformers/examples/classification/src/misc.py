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
    return tokenizer
