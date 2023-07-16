from matplotlib import pyplot as plt
import numpy as np

labels = ['identity_attack', 'profanity', 'severe_toxicity', 'sexually_explicit', 'threat', 'toxicity']
model = ['S','M','L', 'XL', 'LLaMA']

identity_attack_a = [0.5737, 0.5787, 0.6264, 0.61335, 0.71015]
profanity_a = [0.5, 0.5251, 0.57695, 0.66445, 0.81625]
severe_toxicity_a = [0.52435, 0.52465, 0.5626, 0.6176, 0.6439]
sexually_explicit_a = [0.5871, 0.58875, 0.6149, 0.6735, 0.7616]
threat_a = [0.5374, 0.5608, 0.6178, 0.6241, 0.5569]
toxicity_a = [0.5021, 0.54245, 0.64055, 0.68155, 0.7408]

identity_attack_p = [-0.1594538618191228, 0.13297142960619934,0.27401432497431794, 0.14909420024376105, 0.47553865186535615]
profanity_p = [-0.22042233460931795, 0.02270361453908532, 0.18707057949327477, 0.3992726852187268, 0.7116143039634004]
severe_toxicity_p = [-0.16829771644969427,-0.030074577226165043, 0.16056439350248936, 0.3075116177386041, 0.3844732695199883]
sexually_explicit_p = [-0.12117629158050279, 0.0970199674730534, 0.22483737916729016, 0.3901788487240876, 0.5870579945000944]
threat_p = [0.001132733606318824, 0.12555362389505922, 0.23692922150279427, 0.3080139109066506, 0.08290054507179834]
toxicity_p = [-0.1327355773478661, 0.10898989711847147, 0.30628743272528997, 0.4333461831274699, 0.5159369388879256]

attributes_a = [identity_attack_a, profanity_a, severe_toxicity_a, sexually_explicit_a, threat_a, toxicity_a]
attributes_p = [identity_attack_p, profanity_p, severe_toxicity_p, sexually_explicit_p, threat_p, toxicity_p]


avg_a = [0,0,0,0,0]

for i in range(len(attributes_a)):
    for j in range(5):
        avg_a[j]+=attributes_a[i][j]
for j in range(5):
    avg_a[j]/=6

plt.subplot(1, 2, 1) 
plt.plot(model, avg_a, color='black')
for i in range(len(attributes_a)):
    plt.scatter(model, attributes_a[i], label=labels[i])

plt.xlabel('model', fontweight='bold')
plt.ylabel('ACC', fontweight='bold')
plt.legend(fontsize=8)

avg_p = [0,0,0,0,0]

for i in range(len(attributes_p)):
    for j in range(5):
        avg_p[j]+=attributes_p[i][j]
for j in range(5):
    avg_p[j]/=6


plt.subplot(1, 2, 2)
plt.plot(model, avg_p, color='black')
for i in range(len(attributes_p)):
    plt.scatter(model, attributes_p[i], label=labels[i])

plt.xlabel('model', fontweight='bold')
plt.ylabel('PCC', fontweight='bold')

plt.tight_layout()

plt.savefig('image_diagnosis_all.png',bbox_inches='tight')





