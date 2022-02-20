import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from tqdm import tqdm
from collections import Counter
from spacy.training import Alignment

from transformers import AutoTokenizer

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


tokenizer = AutoTokenizer.from_pretrained(
    "/local-scratch1/data/wyshi/privacy/pate/checkpoint/20220211/train10_10epoches/clm_0"
)
sentence = ' Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven " . '
other_tokens = [tokenizer.decode(tok) for tok in tokenizer.encode(sentence)]

doc = nlp(sentence)
spacy_tokens = [x.text for x in doc]

align = Alignment.from_strings(other_tokens, spacy_tokens)
print(f"a -> b, lengths: {align.x2y.lengths}")  # array([1, 1, 1, 1, 1, 1, 1, 1])
print(
    f"a -> b, mapping: {align.x2y.dataXd}"
)  # array([0, 1, 2, 3, 4, 4, 5, 6]) : two tokens both refer to "'s"
print(
    f"b -> a, lengths: {align.y2x.lengths}"
)  # array([1, 1, 1, 1, 2, 1, 1])   : the token "'s" refers to two tokens
print(f"b -> a, mappings: {align.y2x.dataXd}")  # array([0, 1, 2, 3, 4, 5, 6, 7])
