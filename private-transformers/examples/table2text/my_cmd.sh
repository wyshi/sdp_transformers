CUDA_VISIBLE_DEVICES=1 bash table2text/run.sh \
/local-scratch1/data/wyshi/privacy/private-transformers/examples/table2text/output \
/local-scratch1/data/wyshi/privacy/private-transformers/examples/table2text/prefix-tuning \
"e2e"


CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh \
/local-scratch1/data/wyshi/privacy/private-transformers/examples/table2text/output/wiki \
/local-scratch1/data/wyshi/privacy/data \
"wikitext2"



CUDA_VISIBLE_DEVICES=6 bash table2text/run.sh /local-scratch1/data/wyshi/privacy/private-transformers/examples/table2text/output/wiki /local-scratch1/data/wyshi/privacy/data "wikitext2"