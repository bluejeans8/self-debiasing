from matplotlib import pyplot as plt
import numpy as np

labels = ['identity_attack', 'profanity', 'severe_toxicity', 'insult', 'threat', 'toxicity']
status = ['default', 'debiased']

a = [0.10305766038908923, 0.4044576490738832, 0.1718976148167697, 0.28719558194767947, 0.09078083774819609, 0.4832341206215636]

b = [0.05328601964141167, 0.2218957125199827, 0.06462540007106303, 0.1518972512512751, 0.045534652117346974, 0.28393420220365634]

plt.figure(figsize = (10, 5))
plt.xlabel('Attributes')
plt.ylabel('Probability')

idx = np.arange(6)

w = 0.3

plt.bar(idx, a, label=status[0], width=w, color='coral')
plt.bar(idx+w, b, label=status[1], width=w, color='steelblue')
plt.xticks(idx, labels)

plt.legend(fontsize=8, ncol=1)

plt.savefig('image2.png')




