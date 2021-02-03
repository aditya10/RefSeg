import matplotlib.pyplot as plt
import sys
import os

x = []
with open('../ckpts/unc/2/avgloss.txt') as f:
    x = f.readlines()

plt.ylabel('Avg Loss')
plt.xlabel('Iteration')
plt.plot(x, range(0,len(x)))
plt.savefig('./avgloss.png')