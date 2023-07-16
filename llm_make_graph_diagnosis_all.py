from matplotlib import pyplot as plt
import numpy as np

labels = ['identity_attack', 'profanity', 'severe_toxicity', 'sexually_explicit', 'threat', 'toxicity']
model = ['LLaMA','Alpaca']
identity_attack_a = [0.71015, 0.769]
profanity_a = [0.81625, 0.8645]
severe_toxicity_a = [0.6439, 0.762]
sexually_explicit_a = [0.7616, 0.8163]
threat_a = [0.5569, 0.66665]
toxicity_a = [0.7408, 0.83905]

identity_attack_p = [0.47553865186535615, 0.5752357814273312]
profanity_p = [0.7116143039634004, 0.7688875010435394]
severe_toxicity_p = [0.3844732695199883, 0.634940354824032]
sexually_explicit_p = [0.5870579945000944, 0.6685800222491528]
threat_p = [0.08290054507179834, 0.3964257789173413]
toxicity_p = [0.5159369388879256, 0.7013862553477003]

attributes_a = [identity_attack_a, profanity_a, severe_toxicity_a, sexually_explicit_a, threat_a, toxicity_a]
attributes_p = [identity_attack_p, profanity_p, severe_toxicity_p, sexually_explicit_p, threat_p, toxicity_p]

colors = ['red','blue','green','purple','orange','cyan']

avg_a =[0,0]

plt.subplot(1, 2, 1) 

for i in range(6):
    attr = attributes_a[i]
    plt.plot(model, attr, color=colors[i])

for i in range(len(attributes_a)):
    plt.scatter(model, attributes_a[i], label=labels[i], c=colors[i])

plt.xlabel('Model', fontweight='bold')
plt.ylabel('ACC', fontweight='bold')

avg_p = [0,0]

plt.subplot(1, 2, 2) 

for i in range(6):
    attr = attributes_p[i]
    plt.plot(model, attr, color=colors[i])

for i in range(len(attributes_p)):
    plt.scatter(model, attributes_p[i], label=labels[i], c=colors[i])

plt.xlabel('Model', fontweight='bold')
plt.ylabel('PCC', fontweight='bold')
plt.legend(fontsize=8)

plt.tight_layout()

plt.savefig('llm_image_diagnosis_all.png',bbox_inches='tight')





