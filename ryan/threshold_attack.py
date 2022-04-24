import csv
import random
import torch
import torch.nn.functional as F

# read data
data = []
with open("./train_logits.csv", 'r') as f1, open("./non_train_logits.csv", 'r') as f2:
    reader1 = csv.reader(f1)
    reader2 = csv.reader(f2)

    for row in reader1:
        data.append(row)
    for row in reader2:
        data.append(row)

# convert from strings to numeric
l = [int(d[-1]) for d in data]
i = [list(map(float, d[:-1])) for d in data]

# get probs by applying softmax
probs = F.softmax(torch.tensor(i), dim = -1).numpy()

# get classification "confidence" by getting the max probability
confidence = []
for pair in probs:
    confidence.append(max(pair))

# shuffle data
temp = list(zip(l, confidence))
random.shuffle(temp)
labels, conf = zip(*temp)
labels, conf = list(labels), list(conf)

# split data
s = len(labels)
train_labels = labels[:int(s*.9)]
test_labels = labels[int(s*.9):]
train_conf = conf[:int(s*.9)]
test_conf = conf[int(s*.9):]

# calculate threshold
best_acc = -1
best_thresh = -1
for d in range(500,1000):
    total = 0
    total_correct = 0
    acc = -1
    t = d/1000
    for i in range(len(train_labels)):
        if train_conf[i] > t:
            total += 1
            if train_labels[i] == 1:
                total_correct += 1
    
    if total == 0:
        continue

    acc = total_correct/total
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

# calculate test acc
tot = 0
cor = 0
for i in range(len(test_labels)):
    if test_conf[i] > best_thresh:
        tot += 1
        if test_labels[i] == 1:
            cor += 1

print(cor/tot)
print(best_thresh)

