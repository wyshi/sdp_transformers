mkdir -p pate/checkpoint
mkdir -p pate/logs
python -m spacy download en_core_web_sm
mkdir -p private-transformers/examples/table2text/output
gzip -d abcd/data/abcd_v1.1.json.gz