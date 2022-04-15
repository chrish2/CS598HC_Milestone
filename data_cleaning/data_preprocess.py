import random

AE = 'data/DRUG-AE.rel'
DOSE = 'data/DRUG-DOSE.rel'
OUTPUT = 'cleaned.txt'

data = []
def process(path, is_AE):
    with open(path, 'r') as fr:
        for line in fr:
            pubmed_id, text = line.strip().split('|')[:2]
            if is_AE:
                data.append(''.join([text, '>>><<<', str(1)]))
            else:
                data.append(''.join([text, '>>><<<', str(0)]))


process(AE, is_AE=True)
process(DOSE, is_AE=False)

random.shuffle(data)

with open(OUTPUT, 'w') as f:
    for line in data:
        f.write(line+'\n')