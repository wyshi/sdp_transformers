## How to install
\# git clone the repo first 

\# create an environment called "lm_privacy"

conda env create -f environment.yml

\# just in case, permission to execute the file
chmod +x env_transfer.sh

\# create the folders using this script
./env_transfer.sh


## Codebase
The codebase under private-transformers/ is adapted from https://github.com/lxuechen/private-transformers

## Basic procedure
1. Redact the data
2. Fine-tune on the redacted data publicly, and fine-tune on the original data privately

### Data
The redacted data are under data/

The file names indicate the redaction level and percentage of redacted tokens 

https://github.com/wyshi/sdp_transformers/tree/main/scripts shows how to redact


### Fine-tuning
For NLG, the commands are under, https://github.com/wyshi/sdp_transformers/tree/main/private-transformers/examples/table2text/cmds
For classification, the command are under, https://github.com/wyshi/sdp_transformers/tree/main/private-transformers/examples/classification/cmds
