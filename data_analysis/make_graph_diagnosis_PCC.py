from matplotlib import pyplot as plt
import numpy as np

labels = ['identity_attack', 'profanity', 'severe_toxicity', 'sexually_explicit', 'threat', 'toxicity']
model = ['gpt2\n(123M)','gpt2-medium\n(355M)','gpt2-large\n(774M)', 'gpt2-xl\n(1.5B)', 'llama\n(7B)']
identity_attack = [-0.1594538618191228, 0.13297142960619934,0.27401432497431794, 0.14909420024376105, 0.47553865186535615]
profanity = [-0.22042233460931795, 0.02270361453908532, 0.18707057949327477, 0.3992726852187268, 0.7116143039634004]
severe_toxicity = [-0.16829771644969427,-0.030074577226165043, 0.16056439350248936, 0.3075116177386041, 0.3844732695199883]
sexually_explicit = [-0.12117629158050279, 0.0970199674730534, 0.22483737916729016, 0.3901788487240876, 0.5870579945000944]
threat = [0.001132733606318824, 0.12555362389505922, 0.23692922150279427, 0.3080139109066506, 0.34836064951564194]
toxicity = [-0.1327355773478661, 0.10898989711847147, 0.30628743272528997, 0.4333461831274699, 0.5159369388879256]
attributes = [identity_attack, profanity, severe_toxicity, sexually_explicit, threat, toxicity]

avg = [0,0,0,0,0]

f = plt.figure()
ax = f.add_subplot(111)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

for i in range(len(attributes)):
    plt.scatter(model, attributes[i], label=labels[i])
    for j in range(5):
        avg[j]+=attributes[i][j]
for j in range(5):
    avg[j]/=6



plt.plot(model, avg, color='black')
plt.xlabel('model', fontweight='bold')
plt.ylabel('PCC', fontweight='bold')
# plt.legend(fontsize=8)

# plt.scatter(model, identity_attack)
# plt.scatter(model, profanity)
# plt.scatter(model, severe_toxicity)
# plt.scatter(model, sexually_explicit)
# plt.scatter(model, threat)
# plt.scatter(model, toxicity)

plt.savefig('image_PCC_update.png', bbox_inches='tight')

