import argparse
import csv
import copy
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split

# simple binary classifier used for the attack model
class AttackModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AttackModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
      
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


# return the full dataset as well as the labels and inputs individually
def read_data(args):
    data = []
    with open(args.in_train_path, 'r') as f1, open(args.out_train_path, 'r') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        for row in reader1:
            data.append(row)
        for row in reader2:
            data.append(row)

    # convert from strings to numeric
    l = [int(d[-1]) for d in data]
    i = [list(map(float, d[:-1])) for d in data]
    return data, l, i

def prep_data(labels, inputs):
    torch_labels = torch.tensor(labels)
    torch_inputs = torch.tensor(inputs)

    data = TensorDataset(torch_inputs, torch_labels)
    train_size = int(0.80 * len(data))
    dev_size = int(0.1 * len(data))
    test_size = len(data) - train_size - dev_size

    return random_split(data, [train_size, dev_size, test_size])

def compute_accuracy(out, labels):
  pred = out.numpy()
  predictions = [np.argmax(pred[i]) for i in range(len(pred))]

  num_correct = 0
  for i in range(len(predictions)):
    if predictions[i] == labels[i]:
      num_correct += 1

  return num_correct/len(predictions)


def train_model(model, labels, inputs):
    epochs = 1000
    batch_size = 100
    lr = .001

    train_data, dev_data, test_data = prep_data(labels, inputs)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(dataset=train_data, sampler=train_sampler, batch_size=batch_size)

    # unpack dev data since it won't be loaded in batches
    dev_ids = dev_data[:][0]
    dev_labels = dev_data[:][1]

    # parameters for early stopping
    patience = 6
    count=0
    best_acc=-1
    current_acc=-1

    # train model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            outputs = model(x)
            loss = loss_fn(outputs, y)
            
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
    
        # early stopping
        model.eval()
        with torch.no_grad():
            out = model(dev_ids)

        current_acc = compute_accuracy(out, dev_labels)
        print(current_acc)

        if current_acc>best_acc:
            best_model=copy.deepcopy(model)
            best_acc=current_acc
            count=0
        else:
            count+=1
        if count>=patience:
            break 
    
    return best_model, test_data


def main():
    parser = argparse.ArgumentParser()
    # the last three args are required to set the default value
    parser.add_argument("in_train_path", help = "path to the logits of the shadow model's training data",
                        type=str, nargs = "?", const=1, default = "./train_logits.csv")
    parser.add_argument("out_train_path", help = "path to the logits of the shadow model's non-training data",
                        type=str, nargs = "?", const=1, default = "./non_train_logits.csv")
    args = parser.parse_args()

    _, labels, inputs = read_data(args)
    model = AttackModel(len(inputs[0]), 50, 2)
    
    attack_mod, test = train_model(model, labels, inputs)

if __name__ == "__main__":
    main()