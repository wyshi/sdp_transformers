def is_digit(texts_lst):
    """
    texts_lst = ["my", " SS", "N", " is", " 123", "456"]
    return: [0, 0, 0, 0, 1, 1]
    """
    is_private = [int(tok.strip().isdigit()) for tok in texts_lst]
    return is_private