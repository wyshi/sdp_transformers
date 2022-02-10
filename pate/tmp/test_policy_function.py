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
