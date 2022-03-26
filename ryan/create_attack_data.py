import torch
import csv
import numpy as np
from dataclasses import dataclass, field
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import (
    AutoModelForSequenceClassification, 
    AutoConfig, GlueDataset, AutoTokenizer,
    HfArgumentParser, RobertaTokenizer
)
import warnings
warnings.filterwarnings("ignore")

# args for glue dataset
@dataclass
class GlueArgs():
    task_name: str = field(
        metadata = {"help" : "glue task to perform ex. 'sst-2'"}
    )

    data_dir: str = field(
        metadata = {"help" : "directory to datafiles ex. './original/GLUE-SST-2'"}
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        }
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

@dataclass
class ModelArgs():
    model_dir: str = field(
        metadata = {"help" : "Directory to the pretrained shadow model"}
    )

    train: bool = field(
        default = True,
        metadata = {"help" : "Flag to inidicate if the input data is the model's training data or not"}
    )

def compute_accuracy(out, labels):
  pred = out
  predictions = [np.argmax(pred[i]) for i in range(len(pred))]

  num_correct = 0
  for i in range(len(predictions)):
    if predictions[i] == labels[i]:
      num_correct += 1

  return num_correct/len(predictions)

# give the shadow model data and return the output logits
def feed_data(data_slice, model):
    ids = torch.tensor([data_slice[i].input_ids for i in range(len(data_slice))])
    masks = torch.tensor([data_slice[i].attention_mask for i in range(len(data_slice))])
    labels = torch.tensor([data_slice[i].label for i in range(len(data_slice))])

    data = TensorDataset(ids, masks, labels)
    loader = DataLoader(dataset=data, batch_size=32)

    with torch.no_grad():
        final_out = []
        for batch in loader:
            i, m, _ = batch
            out = model(i, token_type_ids=None, attention_mask=m)
            final_out += list(out.logits.numpy())
    
    print(f"Accuracy: {compute_accuracy(final_out, labels):.4f}")
    return final_out

# write shadow model output to csv
def write_data(final_out, model_args):
    if model_args.train:
        attack_dta = [list(final_out[i]) + [1] for i in range(len(final_out))]
        with open('train_logits.csv', 'w') as f:
            write = csv.writer(f)
            write.writerows(attack_dta)
    else:
        attack_dta = [list(final_out[i]) + [0] for i in range(len(final_out))]
        with open('non_train_logits.csv', 'w') as f:
            write = csv.writer(f)
            write.writerows(attack_dta)

def main() :
    parser = HfArgumentParser([GlueArgs, ModelArgs])
    args, model_args = parser.parse_args_into_dataclasses()

    # load models and data
    model_config = AutoConfig.from_pretrained(model_args.model_dir + "/config.json")
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_dir + '/pytorch_model.bin', 
    config=model_config)
    model.eval()

    tokenizer = RobertaTokenizer(model_args.model_dir + '/vocab.json', model_args.model_dir + '/merges.txt')
    train = GlueDataset(args, tokenizer = tokenizer, mode="train")

    # split data 
    size1 = int(.05 * len(train))
    size2 = len(train) - size1
    data_slice, _ = random_split(train, [size1, size2])

    final_out = feed_data(data_slice, model)
    
    write_data(final_out, model_args)
    


if __name__ == "__main__":
    main()