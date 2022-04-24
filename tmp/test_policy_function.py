import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from tqdm import tqdm
from collections import Counter
from spacy.training import Alignment
from collections import defaultdict

from typing import Set, Tuple, Dict, Optional, List
from transformers import AutoTokenizer

import tokenizations

import numpy as np


def get_consecutive_ones_index(bit_vector: List[int]) -> List:
    """get the start and end index for consectuive ones
    [0,1,1,0,0,1] --> [[1,2], [5,5]]
    """
    consecutive_ones_index = []
    prev_is_regular = 0
    i = 0
    while i < len(bit_vector):
        if bit_vector[i] == 0:
            consecutive_ones_index.append([i, i + 1])
            i += 1
            prev_is_regular = 0
        elif bit_vector[i] and prev_is_regular == 0:
            j = i + 1
            while j < len(bit_vector):
                if bit_vector[j]:
                    j += 1
                else:
                    consecutive_ones_index.append([i, j])
                    i = j
                    prev_is_regular = 1
                    break
            if j >= len(bit_vector):
                consecutive_ones_index.append([i, j])
                break
        elif bit_vector[i] and prev_is_regular == 1:
            raise ValueError("you shouldn't be here")

    return consecutive_ones_index


def align_tokens(
    sentence: str,
    tokenizer: AutoTokenizer,
    tokens1: List[str],
    tokensid1: List[int],
    spacy_tokens: List[str],
):
    "align tokens from different tokenizers, tokens1 is from a byte-level tokenizer like BPE"
    "tokens2 is from a regular tokenizer, with space as delimiter"
    is_regular_tokens = [int(tok not in sentence) for tok in tokens1]
    weird_token_ids = get_consecutive_ones_index(is_regular_tokens)
    other_tokens = [tokenizer.decode(tokensid1[i1:i2]) for i1, i2 in weird_token_ids]

    a2b, b2a = tokenizations.get_alignments(other_tokens, spacy_tokens)

    correct_map = []
    for i, interval in enumerate(weird_token_ids):
        for idx in range(interval[0], interval[1]):
            correct_map.append(a2b[i])

    correct_map_b2a = defaultdict(list)
    for i, interval in enumerate(correct_map):
        for j in interval:
            correct_map_b2a[j].append(i)
    import pdb

    pdb.set_trace()
    return correct_map_b2a
    # [for interval in weird_token_ids]
    # [
    #     (tok.strip() in spacy_tokens[idx[-1]])
    #     for tok, idx in zip(other_tokens, a2b)
    #     if idx
    # ]
    # align = Alignment.from_strings(other_tokens, spacy_tokens)

    # ["".join() for idx, y_idx in enumerate(align.x2y.dataXd)]


def main():
    nlp = en_core_web_sm.load()

    # with open("/local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/train.txt") as fh:
    #     texts = fh.readlines()

    # labels = []
    # total = 0
    # for text in tqdm(texts):
    #     doc = nlp(
    #         text
    #         # "European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices"
    #     )
    #     labels.extend([x.label_ for x in doc.ents])
    #     total += len(doc)
    # cnt = Counter(labels)

    tokenizer = AutoTokenizer.from_pretrained(
        "/local-scratch1/data/wyshi/privacy/pate/checkpoint/20220211/train10_10epoches/clm_0"
    )
    sentence = ' Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven " . '
    other_tokens = [tokenizer.decode(tok) for tok in tokenizer.encode(sentence)]
    other_token_ids = tokenizer.encode(sentence)

    doc = nlp(sentence)
    spacy_tokens = [x.text for x in doc]

    align = align_tokens(
        sentence, tokenizer, other_tokens, other_token_ids, spacy_tokens
    )
    # align = Alignment.from_strings(other_tokens, spacy_tokens)
    # print(f"a -> b, lengths: {align.x2y.lengths}")  # array([1, 1, 1, 1, 1, 1, 1, 1])
    # print(
    #     f"a -> b, mapping: {align.x2y.dataXd}"
    # )  # array([0, 1, 2, 3, 4, 4, 5, 6]) : two tokens both refer to "'s"
    # print(
    #     f"b -> a, lengths: {align.y2x.lengths}"
    # )  # array([1, 1, 1, 1, 2, 1, 1])   : the token "'s" refers to two tokens
    # print(f"b -> a, mappings: {align.y2x.dataXd}")  # array([0, 1, 2, 3, 4, 5, 6, 7])


if __name__ == "__main__":
    main()
    print(get_consecutive_ones_index([0, 0, 0, 0, 0]))
    print(get_consecutive_ones_index([0, 1, 1, 0, 0, 1]))
    print(get_consecutive_ones_index([1, 1, 0, 0, 1, 0]))
    while True:
        input("type something to continue")
        bit_vector = np.random.choice(2, size=10)
        print(bit_vector)
        print(get_consecutive_ones_index(bit_vector))
