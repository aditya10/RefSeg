import matplotlib.pyplot as plt
import pandas as pd

y = []
with open('../ckpts/unc/1/avgloss.txt') as f:
    lines = f.readlines()
    y = [float(line.replace("\n", "")) for line in lines]

x = range(1, len(y)*100, 100)

df = pd.Dataframe([x,y])

print(df.head())

plt.ylabel('Avg Acc')
plt.xlabel('Iteration')
plt.plot(x, y)
plt.savefig('./avgacc.png')