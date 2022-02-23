import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from tqdm import tqdm
from collections import Counter
from spacy.training import Alignment

from typing import Set, Tuple, Dict, Optional, List
from transformers import AutoTokenizer
from collections import defaultdict

import tokenizations

import numpy as np

from utils import get_tokens, align_tokens, load_file

NLP = en_core_web_sm.load()

ALL_TYPES = (
    "CARDINAL",
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERSON",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "WORK_OF_ART",
)


def is_digit(token):
    return token.strip().isdigit()


def digit_policy_function(
    tokenizer: AutoTokenizer,
    line: str,
    entity_types: Optional[List] = None,
    debug: Optional[bool] = False,
    return_additional_type_vec: Optional[bool] = False,
):
    original_token_ids, original_tokens = get_tokens(tokenizer, line)
    is_sensitives = [is_digit(tok) for tok in original_tokens]
    return is_sensitives


def ner_policy_function(
    tokenizer: AutoTokenizer,
    line: str,
    entity_types: List,
    debug: Optional[bool] = False,
    return_additional_type_vec: Optional[bool] = False,
):
    """
    import spacy
    nlp = spacy.load("en_core_web_sm")
    nlp.get_pipe("ner").labels
    ('CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART')
    """
    doc = NLP(line)
    original_token_ids, original_tokens = get_tokens(tokenizer, line)
    spacy_tokens = [x.text for x in doc]

    b2a_map = align_tokens(
        line, tokenizer, original_tokens, original_token_ids, spacy_tokens
    )

    ent_to_idx = defaultdict(list)
    for i, x in enumerate(doc):
        if x.ent_type_ in entity_types:
            ent_to_idx[x.ent_type_].append(i)
            if debug:
                try:
                    assert (
                        x.text.strip()
                        in tokenizer.decode(
                            [original_token_ids[_id] for _id in b2a_map[i]]
                        ).strip()
                    )
                except:
                    import pdb

                    pdb.set_trace()

    is_sensitives = np.zeros(len(original_tokens))
    is_sensitives_types = np.zeros(len(original_tokens), dtype=object)
    for ent in ent_to_idx:
        for idx in ent_to_idx[ent]:
            is_sensitives[b2a_map[idx]] = 1
            is_sensitives_types[b2a_map[idx]] = ent

    if return_additional_type_vec:

        return (is_sensitives, is_sensitives_types)
    else:
        return is_sensitives


def main():
    """
    2417786
    {
        0: 1952645,
        "PERSON": 116889,
        "CARDINAL": 32489,
        "NORP": 15794,
        "GPE": 50391,
        "DATE": 71125,
        "ORDINAL": 8340,
        "WORK_OF_ART": 14051,
        "EVENT": 8067,
        "ORG": 99085,
        "FAC": 11120,
        "PRODUCT": 4879,
        "LAW": 1844,
        "LANGUAGE": 452,
        "LOC": 12333,
        "QUANTITY": 8977,
        "MONEY": 3091,
        "TIME": 3088,
        "PERCENT": 3126,
    }
    [
        (0, 0.808),
        ("PERSON", 0.048),
        ("ORG", 0.041),
        ("DATE", 0.029),
        ("GPE", 0.021),
        ("CARDINAL", 0.013),
        ("NORP", 0.007),
        ("WORK_OF_ART", 0.006),
        ("FAC", 0.005),
        ("LOC", 0.005),
        ("QUANTITY", 0.004),
        ("ORDINAL", 0.003),
        ("EVENT", 0.003),
        ("PRODUCT", 0.002),
        ("LAW", 0.001),
        ("MONEY", 0.001),
        ("TIME", 0.001),
        ("PERCENT", 0.001),
        ("LANGUAGE", 0.0),
    ]


    """
    tokenizer = AutoTokenizer.from_pretrained(
        "/local-scratch1/data/wyshi/privacy/pate/checkpoint/20220211/train10_10epoches/clm_0"
    )

    with open("/local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/train.txt") as fh:
        texts = fh.readlines()

    labels = []
    total = 0
    type_cnts = {}
    for text in tqdm(texts):
        is_sensitives, is_sensitives_types = ner_policy_function(
            tokenizer=tokenizer,
            line=text,
            entity_types=ALL_TYPES,
            debug=True,
            return_additional_type_vec=True,
        )
        total += len(is_sensitives)
        cnt = Counter(is_sensitives_types)
        for ent_type in cnt:
            if ent_type not in type_cnts:
                type_cnts[ent_type] = cnt[ent_type]
            else:
                type_cnts[ent_type] += cnt[ent_type]
    print(total)
    print(type_cnts)


if __name__ == "__main__":
    main()
    tokenizer = AutoTokenizer.from_pretrained(
        "/local-scratch1/data/wyshi/privacy/pate/checkpoint/20220211/train10_10epoches/clm_0"
    )

    sentence = ' Krasnyi Kavkaz ( from Russian : " Красный Кавказ " - " Red Caucasus " ) was a cruiser of the Soviet Navy that began construction during World War I , but was still incomplete during the Russian Revolution . Her design was heavily modified by the Soviets and she was completed in 1932 . During World War II she supported Soviet troops during the Siege of Odessa , Siege of Sevastopol , and the Kerch @-@ Feodosiya Operation in the winter of 1941 — 42 . She was awarded the Guards title on 3 April 1942 . She was reclassified as a training ship in May 1947 before being used as a target in 1952 .'

    is_sensitives = ner_policy_function(
        tokenizer=tokenizer, line=sentence, entity_types=["PERSON"], debug=True
    )
