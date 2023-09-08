import os
from nltk.tokenize import sent_tokenize

FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/wikitext-2-raw/train.txt")
DIR = os.path.dirname(os.path.abspath(__file__))

with open(FILE) as inp:
    txt = inp.readlines()

def wiki(wiki):
    wiki_dedup = []
    unique_sents = set()

    for i, line in enumerate(wiki):
        sents = sent_tokenize(line)
        for i, sent in enumerate(sents):
            if sent in unique_sents:
                sents[i] = '<MASK>'
            else:
                unique_sents.add(sent)
        final_sent = ' '.join(sents)
        wiki_dedup.append(final_sent)
    
    return wiki_dedup

wiki_dedup = wiki(txt)

with open(os.path.join(DIR, 'train.txt'), 'w') as inp:
    for l in wiki_dedup:
        inp.write(l+"\n")

# def abcd_nltk(abcd):
#     abcd_dedup = []
#     unique_sents = set()

#     for i, line in enumerate(abcd):
#         if len(line.split(':')) == 2:
#             usr, text = line.split(':')
#             sents = sent_tokenize(text)
#         elif len(line.split(':')) > 2:
#             usr = line.split(':')[0]
#             text = ':'.join(line.split(':')[1:])
#             sents = sent_tokenize(text)
#         else:
#             sents = sent_tokenize(line)
        
#         if len(sents) > 0:
#             if sents[-1] == '!':
#                 sents.pop(-1)
#                 sents[-1] = sents[-1] + '!'
#             elif sents[-1] == '?':
#                 sents.pop(-1)
#                 sents[-1] = sents[-1] + '?'
        
#         for i, sent in enumerate(sents):
#             if sent in unique_sents:
#                 sents[i] = '<MASK>.'
#             else:
#                 unique_sents.add(sent)
        
#         final_sent = ' '.join(sents)
#         if len(line.split(':')) >= 2:
#             abcd_dedup.append(usr + ':' + final_sent)
#         else:
#             abcd_dedup.append(final_sent)
            
#     return abcd_dedup
