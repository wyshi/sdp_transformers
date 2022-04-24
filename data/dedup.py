from nltk.tokenize import sent_tokenize

with open("./abcd/abcd_original/train.txt") as inp:
    txt = inp.readlines()

def wiki(wiki):
    wiki_dedup = []
    unique_sents = set()

    for i, line in enumerate(wiki):
        sents = sent_tokenize(line)
        for i, sent in enumerate(sents):
            if sent in unique_sents:
                sents[i] = '<MASK>'
                print('sentence masked')
            else:
                unique_sents.add(sent)
        final_sent = ' '.join(sents)
        wiki_dedup.append(final_sent)
    
    return wiki_dedup

def abcd_nltk(abcd):
    abcd_dedup = []
    unique_sents = set()

    for i, line in enumerate(abcd):
        if len(line.split(':')) == 2:
            usr, text = line.split(':')
            sents = sent_tokenize(text)
        else:
            sents = sent_tokenize(line)
        
        if len(sents) > 0:
            if sents[-1] == '!':
                sents.pop(-1)
                sents[-1] = sents[-1] + '!'
            elif sents[-1] == '?':
                sents.pop(-1)
                sents[-1] = sents[-1] + '?'
        
        for i, sent in enumerate(sents):
            if sent in unique_sents:
                sents[i] = '<MASK>'
                print('sentence masked')
            else:
                unique_sents.add(sent)
        
        final_sent = ' '.join(sents)
        if len(line.split(':')) == 2:
            abcd_dedup.append(usr + ':' + final_sent)
        else:
            abcd_dedup.append(final_sent)
            
    return abcd_dedup

abcd_dedup = abcd_nltk(txt)

with open('./train.txt', 'w') as inp:
    for l in abcd_dedup:
        inp.write(l+"\n")


# with open("./abcd/abcd_original/train.txt") as inp:
#     wiki = inp.readlines()

# wiki_dedup = []
# unique_sents = set()
# for i, line in enumerate(wiki):
#     sents = sent_tokenize(line)
#     for i, sent in enumerate(sents):
#         if sent in unique_sents:
#             sents[i] = '<MASK>'
#             print('sentence masked')
#         else:
#             unique_sents.add(sent)
#     final_sent = ' '.join(sents)
#     wiki_dedup.append(final_sent)

# with open('./train.txt', 'w') as inp:
#     for l in wiki_dedup:
#         inp.write(l+"\n")

# with open("./abcd/abcd_original/train.txt") as inp:
#     wiki = inp.readlines()