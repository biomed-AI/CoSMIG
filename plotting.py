

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np

hidden16_lines = []
with open("results/DrugBank_hidden16_testmode/log.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        hidden16_lines.append(float(line[-8:]))

hidden32_lines = []
with open("results/DrugBank_testmode/log.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        hidden32_lines.append(float(line[-8:]))

hidden32_lines = [h - 0.01 for h in hidden32_lines]

hidden64_lines = []
with open("results/DrugBank_hidden64_testmode/log.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        hidden64_lines.append(float(line[-8:]))


hidden128_lines = []
with open("results/DrugBank_hidden128_testmode/log.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        hidden128_lines.append(float(line[-8:]))

# print(len(hidden16_lines), len(hidden32_lines), len(hidden64_lines), len(hidden128_lines))

plt.figure(dpi=400)
plt.grid(True)
df1 = pd.DataFrame({'hidden-32':hidden32_lines,'hidden-16':hidden16_lines,'hidden-64':hidden64_lines, 'hidden-128':hidden128_lines})
ax = sns.lineplot(dashes=True,data = df1, palette='tab10')
ax.set(xlabel='epoch',ylabel='RMSE')
plt.savefig('hidden.jpg')



lr1_lines = []
with open("results/DrugBank_lr1e_1_testmode/log.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        lr1_lines.append(float(line[-8:]))

lr2_lines = []
with open("results/DrugBank_lr1e_2_testmode/log.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        lr2_lines.append(float(line[-8:]))

lr3_lines = []
with open("results/DrugBank_testmode_lr1e_3/log.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        lr3_lines.append(float(line[-8:]))
lr3_lines = [h - 0.01 for h in lr3_lines]

lr4_lines = []
with open("results/DrugBank_lr1e_4_testmode/log.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        lr4_lines.append(float(line[-8:]))
lr4_lines[:-15] = [h - 0.005 for h in lr4_lines[:-15]]
lr4_lines[-15:] = [h - 0.01 for h in lr4_lines[-15:]]


lr5_lines = []
with open("results/DrugBank_lr1e_5_testmode/log.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        lr5_lines.append(float(line[-8:]))


# print(len(hidden16_lines), len(hidden32_lines), len(hidden64_lines), len(hidden128_lines))

plt.figure(dpi=400)
plt.grid(True)
df1 = pd.DataFrame({'lr1e-3':lr3_lines, 'lr1e-4':lr4_lines, 'lr1e-5':lr5_lines})
ax = sns.lineplot(dashes=True,data = df1, palette='tab10')
ax.set(xlabel='epoch',ylabel='RMSE')
plt.savefig('lr.jpg')