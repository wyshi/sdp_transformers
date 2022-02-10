import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from tqdm import tqdm
from collections import Counter
from spacy.training import Alignment

nlp = en_core_web_sm.load()

with open("/local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/train.txt") as fh:
    texts = fh.readlines()

labels = []
total = 0
for text in tqdm(texts):
    doc = nlp(
        text
        # "European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices"
    )
    labels.extend([x.label_ for x in doc.ents])
    total += len(doc)
cnt = Counter(labels)



other_tokens = ["i", "listened", "to", "obama", "'", "s", "podcasts", "."]
spacy_tokens = ["i", "listened", "to", "obama", "'s", "podcasts", "."]
align = Alignment.from_strings(other_tokens, spacy_tokens)
print(f"a -> b, lengths: {align.x2y.lengths}")  # array([1, 1, 1, 1, 1, 1, 1, 1])
print(f"a -> b, mapping: {align.x2y.dataXd}")  # array([0, 1, 2, 3, 4, 4, 5, 6]) : two tokens both refer to "'s"
print(f"b -> a, lengths: {align.y2x.lengths}")  # array([1, 1, 1, 1, 2, 1, 1])   : the token "'s" refers to two tokens
print(f"b -> a, mappings: {align.y2x.dataXd}")  # array([0, 1, 2, 3, 4, 5, 6, 7])