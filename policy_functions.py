from utils import get_tokens


def is_digit(token):
    return token.strip().isdigit()


def digit_policy_function(tokenizer, line):
    original_input_ids, original_tokens = get_tokens(tokenizer, line)
    is_sensitives = [is_digit(tok) for tok in original_tokens]
    return is_sensitives
