import torch.nn as nn

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
from spacy.matcher import Matcher

import tokenizations

import numpy as np

# def normalize_sentence(original_sentence, is_sensitives_types):
MAP = {"agent": "SYS:", "customer": "USR:", "action": "ACT:"}

EOS = "<|endoftext|>"


def get_tokens(tokenizer, line):
    original_input_ids = tokenizer.encode(line)
    original_tokens = [
        tokenizer.decode(input_id, clean_up_tokenization_spaces=False) for input_id in original_input_ids
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
    return correct_map_b2a


def convert_abcd_line(speaker: str, utt: str):
    return MAP[speaker] + utt + "\n\n"


def get_entities(sent, nlp):
    # modified from https://hami-asmai.medium.com/relationship-extraction-from-any-web-articles-using-spacy-and-jupyter-notebook-in-6-steps-4444ee68763f
    ## chunk 1
    # ent1 = ""
    # ent2 = ""

    ent1s = []
    ent2s = []

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""

    #############################################################

    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                ent1s.append(ent1)
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

            ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text
                ent2s.append(ent2)

            ## chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    #############################################################

    return [ent1.strip() for ent1 in ent1s], [ent2.strip() for ent2 in ent2s]  # [ent1.strip(), ent2.strip()]


def get_relation(sent, nlp):
    # modified from https://hami-asmai.medium.com/relationship-extraction-from-any-web-articles-using-spacy-and-jupyter-notebook-in-6-steps-4444ee68763f

    doc = nlp(sent)

    # Matcher class object
    matcher = Matcher(nlp.vocab)

    # define the pattern
    pattern = [{"DEP": "ROOT"}, {"DEP": "prep", "OP": "?"}, {"DEP": "agent", "OP": "?"}, {"POS": "ADJ", "OP": "?"}]

    matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    # k = len(matches) - 1

    spans = [doc[matches[i][1] : matches[i][2]] for i in range(len(matches))]

    return [span.text for span in spans]


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")  # xx_ent_wiki_sm,xx_sent_ud_sm,en_core_web_sm
    sent = " The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n . "
    entity = get_entities(sent, nlp)

    relation = get_relation(sent, nlp)
    print(entity)
    print(relation)
    import pdb

    pdb.set_trace()
