import random

AE = '../data/DRUG-AE.rel'
NEG = '../data/ADE-NEG.txt'
OUTPUT = 'cleaned.txt'

data = []

with open(AE, 'r') as fr:
    for line in fr:
        pubmed_id, text = line.strip().split('|')[:2]
        data.append(''.join([text, '>>><<<', str(1)]))

with open(NEG, 'r') as fr:
    for line in fr:
        pubmed_id, neg = line.strip().split(' ')[:2]
        text = ' '.join(line.strip().split(' ')[2:])
        data.append(''.join([text, '>>><<<', str(0)]))

random.shuffle(data)

with open(OUTPUT, 'w') as f:
    for line in data:
        f.write(line+'\n')