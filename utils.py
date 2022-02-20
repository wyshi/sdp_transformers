import torch.nn as nn


def get_tokens(tokenizer, line):
    original_input_ids = tokenizer.encode(line)
    original_tokens = [
        tokenizer.decode(input_id, clean_up_tokenization_spaces=False)
        for input_id in original_input_ids
    ]
    return original_input_ids, original_tokens


def load_file(
    file_dir="/local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/train.txt",
):
    with open(
        file_dir,
        "r",
        encoding="utf8",
    ) as f:
        lines = f.readlines()

    return lines


def predict_with_model(
    tokenizer,
    beam_size,
    model,
    input_ids,
    i_th_token,
    limit_to_sensitive,
    public_token_ids,
):
    if limit_to_sensitive:
        assert public_token_ids is not None
    else:
        assert public_token_ids is None

    outputs = model.generate(
        input_ids=input_ids,
        max_length=i_th_token + 1 if input_ids is not None else 2,
        num_return_sequences=beam_size,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        num_beams=beam_size,
        bad_words_ids=public_token_ids,
    )
    pred_token_ids = [output[-1].item() for output in outputs.sequences]
    pred_token_scores = nn.functional.softmax(
        outputs.scores[0][0],
    )
    return pred_token_ids, pred_token_scores
