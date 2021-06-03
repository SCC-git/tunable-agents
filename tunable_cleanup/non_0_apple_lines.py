filename = './logs/2021-6-1_11_8/output.txt'

with open(filename) as f:
    content = f.readlines()

for line in content:
    line = line.strip().split()
    if len(line) > 0 and line[-6] != '0,':
        print(' '.join(line))