from matplotlib import pyplot as plt
import numpy as np

labels = ['identity_attack', 'profanity', 'severe_toxicity', 'insult', 'threat', 'toxicity']
model = ['llama\n(7B)','alpaca\n(7B)']

### default graph

llama_default = {'IDENTITY_ATTACK': 0.10890737429747684, 'SEVERE_TOXICITY': 0.18098412023007546, 'INSULT': 0.31902926002523085, 'THREAT': 0.08965738308292678, 'PROFANITY': 0.4596289532973087, 'TOXICITY': 0.5319740634268294}

alpaca_default = {'THREAT': 0.07186597245062759, 'TOXICITY': 0.441060850987531, 'INSULT': 0.2567934916158156, 'SEVERE_TOXICITY': 0.12944970269831801, 'IDENTITY_ATTACK': 0.0894245378933892, 'PROFANITY': 0.36855238881966573}

identity_attack_1 = [llama_default['IDENTITY_ATTACK'], alpaca_default['IDENTITY_ATTACK'] ]
profanity_1 = [llama_default['PROFANITY'],alpaca_default['PROFANITY']]
severe_toxicity_1 = [llama_default['SEVERE_TOXICITY'],alpaca_default['SEVERE_TOXICITY']]
insult_1 = [llama_default['INSULT'],alpaca_default['INSULT']]
threat_1 = [llama_default['THREAT'],alpaca_default['THREAT']]
toxicity_1 = [llama_default['TOXICITY'], alpaca_default['TOXICITY']]


attributes_1 = [identity_attack_1, profanity_1, severe_toxicity_1, insult_1, threat_1, toxicity_1]
avg_1 = [0,0]

for i in range(len(attributes_1)):
    # plt.scatter(model, attributes_1[i], label=labels[i])
    for j in range(2):
        avg_1[j]+=attributes_1[i][j]
for j in range(2):
    avg_1[j]/=6

# plt.plot(model, avg_1, color='blue')


### debiasing graph

llama_debiased = {'THREAT': 0.046794640581810584, 'SEVERE_TOXICITY': 0.05974380558093047, 'IDENTITY_ATTACK': 0.06040582507524734, 'INSULT': 0.1625439865500417, 'TOXICITY': 0.2937590617409051, 'PROFANITY': 0.2182627748977369}

alpaca_debiased = {'PROFANITY': 0.21135535097897384, 'SEVERE_TOXICITY': 0.05357531011504626, 'IDENTITY_ATTACK': 0.054143277344407145, 'THREAT': 0.03980883639730868, 'INSULT': 0.1494949636384358, 'TOXICITY': 0.2777539231289323}


identity_attack_2 = [llama_debiased['IDENTITY_ATTACK'], alpaca_debiased['IDENTITY_ATTACK'] ]
profanity_2 = [llama_debiased['PROFANITY'],alpaca_debiased['PROFANITY']]
severe_toxicity_2 = [llama_debiased['SEVERE_TOXICITY'],alpaca_debiased['SEVERE_TOXICITY']]
insult_2 = [llama_debiased['INSULT'],alpaca_debiased['INSULT']]
threat_2 = [llama_debiased['THREAT'],alpaca_debiased['THREAT']]
toxicity_2 = [llama_debiased['TOXICITY'], alpaca_debiased['TOXICITY']]

attributes_2 = [identity_attack_2, profanity_2, severe_toxicity_2, insult_2, threat_2, toxicity_2]
avg_2 = [0,0]

for i in range(len(attributes_2)):
    # plt.scatter(model, attributes_2[i], label=labels[i])
    for j in range(2):
        avg_2[j]+=attributes_2[i][j]
for j in range(2):
    avg_2[j]/=6

# plt.plot(model, avg_2, color='red')

######

# diff_abs = [0,0,0,0,0]
# diff_rel = [0,0,0,0,0]

# for i in range(5):
#     diff_abs[i] = avg_1[i] - avg_2[i]

# for i in range(5):
#     diff_rel[i] = (avg_1[i] - avg_2[i]) / avg_1[i]

# print("diff_abs:", diff_abs)
# print("diff_rel:", diff_rel)

######

# for j in range(len(attributes_1)):

#     diff_abs = [0,0,0,0,0]
#     diff_rel = [0,0,0,0,0]

#     for i in range(5):
#         diff_abs[i] = attributes_1[j][i] - attributes_2[j][i]

#     for i in range(5):
#         diff_rel[i] = (attributes_1[j][i] - attributes_2[j][i]) / attributes_1[j][i]

#     print(labels[j])
#     print("diff_abs:", diff_abs)
#     print("diff_rel:", diff_rel)

######




plt.figure(figsize = (10, 5))
plt.xlabel('model', fontweight='bold')
plt.ylabel('average probabilities', fontweight='bold')
# plt.legend(fontsize=8)

idx = np.arange(2)

w = 0.3

plt.bar(idx-0.5*w, avg_1, label='default', width=w, color='#FFB6C1')
plt.bar(idx+0.5*w, avg_2, label='debiased', width=w, color='#B0E0E6')
plt.xticks(idx, model)

plt.legend(fontsize=8, ncol=1)



plt.savefig('llm_image_debiasing_rate.png',bbox_inches='tight')

