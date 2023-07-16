from matplotlib import pyplot as plt
import numpy as np

labels = ['identity_attack', 'profanity', 'severe_toxicity', 'sexually_explicit', 'threat', 'toxicity']
model = ['llama\n(7B)','alpaca\n(7B)']
identity_attack = [0.71015, 0.769]
profanity = [0.81625, 0.8645]
severe_toxicity = [0.6439, 0.762]
sexually_explicit = [0.7616, 0.8163]
threat = [0.5569, 0.66665]
toxicity = [0.7408, 0.83905]
attributes = [identity_attack, profanity, severe_toxicity, sexually_explicit, threat, toxicity]

avg = [0,0]
colors = ['red','blue','green','purple','orange','cyan']

for i in range(len(attributes)):
    plt.scatter(model, attributes[i], label=labels[i], c=colors[i])
    for j in range(2):
        avg[j]+=attributes[i][j]
for j in range(2):
    avg[j]/=6


for i in range(6):
    attr = attributes[i]
    plt.plot(model, attr, color=colors[i])


plt.xlabel('Model', fontweight='bold')
plt.ylabel('ACC', fontweight='bold')
plt.legend(fontsize=8)

# plt.scatter(model, identity_attack)
# plt.scatter(model, profanity)
# plt.scatter(model, severe_toxicity)
# plt.scatter(model, sexually_explicit)
# plt.scatter(model, threat)
# plt.scatter(model, toxicity)

plt.savefig('llm_image_ACC.png',bbox_inches='tight')

