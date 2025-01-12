from matplotlib import pyplot as plt
import numpy as np

labels = ['identity_attack', 'profanity', 'severe_toxicity', 'sexually_explicit', 'threat', 'toxicity']
model = ['gpt2\n(123M)','gpt2-medium\n(355M)','gpt2-large\n(774M)', 'gpt2-xl\n(1.5B)', 'llama\n(7B)']
identity_attack = [0.5737, 0.5787, 0.6264, 0.61335, 0.71015]
profanity = [0.5, 0.5251, 0.57695, 0.66445, 0.81625]
severe_toxicity = [0.52435, 0.52465, 0.5626, 0.6176, 0.6439]
sexually_explicit = [0.5871, 0.58875, 0.6149, 0.6735, 0.7616]
threat = [0.5374, 0.5608, 0.6178, 0.6241, 0.64705]
toxicity = [0.5021, 0.54245, 0.64055, 0.68155, 0.7408]
attributes = [identity_attack, profanity, severe_toxicity, sexually_explicit, threat, toxicity]

avg = [0,0,0,0,0]

for i in range(len(attributes)):
    plt.scatter(model, attributes[i], label=labels[i])
    for j in range(5):
        avg[j]+=attributes[i][j]
for j in range(5):
    avg[j]/=6


plt.plot(model, avg, color='black')
plt.xlabel('model', fontweight='bold')
plt.ylabel('ACC', fontweight='bold')
plt.legend(fontsize=8)

# plt.scatter(model, identity_attack)
# plt.scatter(model, profanity)
# plt.scatter(model, severe_toxicity)
# plt.scatter(model, sexually_explicit)
# plt.scatter(model, threat)
# plt.scatter(model, toxicity)

plt.savefig('image_ACC_update.png',bbox_inches='tight')

