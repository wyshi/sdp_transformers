CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
/local-scratch1/data/wyshi/privacy/private-transformers/examples/table2text/output \
/local-scratch1/data/wyshi/privacy/private-transformers/examples/table2text/prefix-tuning \
"e2e"


CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
/local-scratch1/data/wyshi/privacy/private-transformers/examples/table2text/output/wiki \
/local-scratch1/data/wyshi/privacy/data \
"wikitext2"


# public
CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/wiki_public \
../../data \
"wikitext2" \
"gpt2" \
3 \
yes \
yes


# sanitization, low
CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/wiki_person \
../../data \
"wikitext2-delex-person" \
"gpt2" \
3 \
yes \
yes

# sanitization, medium
CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/wiki_medium \
../../data \
"wikitext2-delex-medium" \
"gpt2" \
3 \
yes \
yes

# sanitization, high
CUDA_VISIBLE_DEVICES=5 bash table2text/run.sh \
table2text/output/wiki_high \
../../data \
"wikitext2-delex-high" \
"gpt2" \
3 \
yes \
yes


# finetune, low
CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/wiki_person_sdp_finetune \
../../data \
"wikitext2" \
"table2text/output/wiki_person/checkpoint-2298" \
3 \
yes \
no \
yes

# finetune, medium
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/wiki_medium_sdp_finetune \
../../data \
"wikitext2" \
"table2text/output/wiki_medium/checkpoint-1127" \
3 \
yes \
no \
yes

# finetune, high
CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/wiki_high_sdp_finetune \
../../data \
"wikitext2" \
"table2text/output/wiki_high/checkpoint-1125" \
3 \
yes \
no \
yes


# dpsgd
CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
table2text/output/wiki_dpsgd \
../../data \
"wikitext2" \
"gpt2" \
3 \
yes \
no \
no \
15



###### abcd
# public
CUDA_VISIBLE_DEVICES=0 bash table2text/run.sh \
table2text/output/abcd_public \
../../data \
"wikitext2-abcd" \
"gpt2" \
3 \
yes \
yes \
no \
10


# sanitization, low
CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
table2text/output/abcd_delex \
../../data \
"wikitext2-abcd-delex" \
"gpt2" \
3 \
yes \
yes \
no \
10

# dpsgd
CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/abcd_dpsgd \
../../data \
"wikitext2-abcd" \
"gpt2" \
3 \
yes \
no \
no \
30


# finetune, abcd
CUDA_VISIBLE_DEVICES=2 bash table2text/run.sh \
table2text/output/abcd_sdp_finetune \
../../data \
"wikitext2-abcd" \
"table2text/output/abcd_delex/checkpoint-1033" \
3 \
yes \
no \
yes \
10
