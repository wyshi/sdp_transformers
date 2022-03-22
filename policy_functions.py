import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from tqdm import tqdm
from collections import Counter
from spacy.training import Alignment
from allennlp.predictors.predictor import Predictor

from typing import Set, Tuple, Dict, Optional, List
from transformers import AutoTokenizer
from collections import defaultdict

import tokenizations

import numpy as np

from utils import get_tokens, align_tokens, load_file, ALL_TYPES, get_special_tokens, SPECIAL_TOKENS_MAP

NLP = en_core_web_sm.load()


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


def get_spacy_tokens_and_doc(line):
    doc = NLP(line)
    spacy_tokens = [x.text for x in doc]
    return spacy_tokens, doc


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
    original_token_ids, original_tokens = get_tokens(tokenizer, line)
    spacy_tokens, doc = get_spacy_tokens_and_doc(line)

    b2a_map = align_tokens(line, tokenizer, original_tokens, original_token_ids, spacy_tokens)

    interval_to_sensitive_type_dict = {}
    ent_to_idx = defaultdict(list)
    for i, x in enumerate(doc):
        if x.ent_type_ in entity_types:
            ent_to_idx[x.ent_type_].append(i)
            interval_to_sensitive_type_dict[tuple(b2a_map[i])] = get_special_tokens(x.ent_type_)
            if debug:
                try:
                    assert x.text.strip() in tokenizer.decode([original_token_ids[_id] for _id in b2a_map[i]]).strip()
                except:
                    import pdb

                    pdb.set_trace()

        else:
            interval_to_sensitive_type_dict[tuple(b2a_map[i])] = 0

    is_sensitives = np.zeros(len(original_tokens))
    is_sensitives_types = np.zeros(len(original_tokens), dtype=object)
    for ent in ent_to_idx:
        for idx in ent_to_idx[ent]:
            is_sensitives[b2a_map[idx]] = 1
            is_sensitives_types[b2a_map[idx]] = ent

    import pdb

    pdb.set_trace()

    if return_additional_type_vec:
        return (is_sensitives, is_sensitives_types, interval_to_sensitive_type_dict)
    else:
        return is_sensitives


def delex_line(
    line: str,
    entity_types: List,
    return_stat: Optional[bool] = False,
    dep_types: Optional[list] = None,
    pos_types: Optional[list] = None,
    predictor=None,
    use_single_mask_token=True,
    concat_consecutive_special_tokens=True,
):
    if line.endswith("\n"):
        endswith_new_line = True
        line = line[:-1]
        assert not line.endswith("\n"), "line still ends with \n"
    else:
        endswith_new_line = False
    _, doc = get_spacy_tokens_and_doc(line)
    words = [tok.text for tok in doc]
    spaces = [True if tok.whitespace_ else False for tok in doc]

    # SRL
    if predictor:
        predictions = predictor.predict(sentence=line)
        other_tokens = predictions["words"]
        a2b, b2a = tokenizations.get_alignments(other_tokens, words)
        predicate_original_indexes = [p["tags"].index("B-V") for p in predictions["verbs"]]
        predicate_spacy_indexes = []
        for idx in predicate_original_indexes:
            predicate_spacy_indexes.extend(a2b[idx])

    # delex
    delexed = 0
    for i, x in enumerate(doc):
        need_to_add = False
        if predictor:
            # SRL
            if i in predicate_spacy_indexes:
                words[i] = get_special_tokens("pred", use_single_mask_token)
                need_to_add = True
        if x.ent_type_ in entity_types:
            # named entity
            words[i] = get_special_tokens(x.ent_type_, use_single_mask_token)
            need_to_add = True
        if dep_types:
            # dep parser
            for dep_type_ in dep_types:
                if dep_type_ in x.dep_.lower():
                    words[i] = get_special_tokens(dep_type_.upper(), use_single_mask_token)
                    need_to_add = True
        if pos_types:
            # pos tag
            if x.pos_ in pos_types:
                words[i] = get_special_tokens(x.pos_, use_single_mask_token)
                need_to_add = True
        if need_to_add:
            delexed += 1
    total = len(doc)

    # rejoin them
    doc2 = spacy.tokens.doc.Doc(NLP.vocab, words=words, spaces=spaces)

    return_text = doc2.text
    if endswith_new_line:
        return_text = return_text + "\n"
    if concat_consecutive_special_tokens:
        all_special_tokens = list(SPECIAL_TOKENS_MAP.values())
        tokens = return_text.split(" ")
        post_tokens = []
        prev_token = None
        for tok in tokens:
            if tok in all_special_tokens and tok == prev_token:
                continue
            post_tokens.append(tok)
            prev_token = tok
        return_text = " ".join(post_tokens)
    if return_stat:
        return return_text, delexed, total
    else:
        return return_text


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
        (is_sensitives, is_sensitives_types, interval_to_sensitive_type_dict,) = ner_policy_function(
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
    # main()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # sentence = ' Krasnyi Kavkaz ( from Russian : " Красный Кавказ " - " Red Caucasus " ) was a cruiser of the Soviet Navy that began construction during World War I , but was still incomplete during the Russian Revolution . Her design was heavily modified by the Soviets and she was completed in 1932 . During World War II she supported Soviet troops during the Siege of Odessa , Siege of Sevastopol , and the Kerch @-@ Feodosiya Operation in the winter of 1941 — 42 . She was awarded the Guards title on 3 April 1942 . She was reclassified as a training ship in May 1947 before being used as a target in 1952 .'

    sentence = "Can I please borrow 50000 dollars from you to buy some Microsoft stock?"
    (is_sensitives, is_sensitives_types, interval_to_sensitive_type_dict,) = ner_policy_function(
        tokenizer=tokenizer,
        line=sentence,
        entity_types=ALL_TYPES,
        debug=True,
        return_additional_type_vec=True,
    )
    import pdb

    pdb.set_trace()
