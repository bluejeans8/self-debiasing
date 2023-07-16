from matplotlib import pyplot as plt
import numpy as np

labels = ['identity_attack', 'profanity', 'severe_toxicity', 'sexually_explicit', 'threat', 'toxicity']
model = ['llama\n(7B)','alpaca\n(7B)']
identity_attack = [0.47553865186535615, 0.5752357814273312]
profanity = [0.7116143039634004, 0.7688875010435394]
severe_toxicity = [0.3844732695199883, 0.634940354824032]
sexually_explicit = [0.5870579945000944, 0.6685800222491528]
threat = [0.08290054507179834, 0.3964257789173413]
toxicity = [0.5159369388879256, 0.7013862553477003]
attributes = [identity_attack, profanity, severe_toxicity, sexually_explicit, threat, toxicity]

avg = [0,0]
colors = ['red','blue','green','purple','orange','cyan']


f = plt.figure()
ax = f.add_subplot(111)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

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
plt.ylabel('PCC', fontweight='bold')
# plt.legend(fontsize=8)

# plt.scatter(model, identity_attack)
# plt.scatter(model, profanity)
# plt.scatter(model, severe_toxicity)
# plt.scatter(model, sexually_explicit)
# plt.scatter(model, threat)
# plt.scatter(model, toxicity)

plt.savefig('llm_image_PCC.png', bbox_inches='tight')

