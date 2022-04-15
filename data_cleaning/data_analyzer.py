
import os
pos_path = 'DRUG-DOSE.rel'


arr = []
max = 0
with open(pos_path, 'r') as fr:
    for line in fr:
        pubmed_id, text = line.strip().split('|')[:2]
        length = len(text.split())
        arr.append(length)
        if length > max:
            max = length

avg  = sum(arr) / len(arr)
print('avg', avg)
print('max', max)