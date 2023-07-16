from matplotlib import pyplot as plt
import numpy as np

labels = ['identity_attack', 'profanity', 'severe_toxicity', 'insult', 'threat', 'toxicity']
model = ['gpt2\n(123M)','gpt2-medium\n(355M)','gpt2-large\n(774M)', 'gpt2-xl\n(1.5B)', 'llama\n(7B)']

### default graph

gpt2_default = {'IDENTITY_ATTACK': 0.08315541188204688, 'TOXICITY': 0.4485822675073718, 'THREAT': 0.0823593248088465, 'SEVERE_TOXICITY': 0.15743747438930586, 'PROFANITY': 0.3667531446214228, 'INSULT': 0.256210519390633}
gpt2_medium_default = {'THREAT': 0.08934407417377464, 'TOXICITY': 0.47612812345657757, 'PROFANITY': 0.3935311064036975, 'IDENTITY_ATTACK': 0.09564739933465169, 'INSULT': 0.27752067724677537, 'SEVERE_TOXICITY': 0.16925201249766123}
gpt2_large_default = {'INSULT': 0.2844179486563091, 'SEVERE_TOXICITY': 0.17219292188499552, 'TOXICITY': 0.4790429840612015, 'IDENTITY_ATTACK': 0.10507169643070381, 'PROFANITY': 0.39955315361073007, 'THREAT': 0.08403271681716733}
gpt2_xl_default = {'IDENTITY_ATTACK': 0.10305766038908923, 'THREAT': 0.09078083774819609, 'PROFANITY': 0.4044576490738832, 'SEVERE_TOXICITY': 0.1718976148167697, 'TOXICITY': 0.4832341206215636, 'INSULT': 0.28719558194767947}
llama_default = {'IDENTITY_ATTACK': 0.10890737429747684, 'SEVERE_TOXICITY': 0.18098412023007546, 'INSULT': 0.31902926002523085, 'THREAT': 0.08965738308292678, 'PROFANITY': 0.4596289532973087, 'TOXICITY': 0.5319740634268294}


identity_attack_1 = [gpt2_default['IDENTITY_ATTACK'],gpt2_medium_default['IDENTITY_ATTACK'],gpt2_large_default['IDENTITY_ATTACK'],gpt2_xl_default['IDENTITY_ATTACK'],llama_default['IDENTITY_ATTACK']]
profanity_1 = [gpt2_default['PROFANITY'],gpt2_medium_default['PROFANITY'],gpt2_large_default['PROFANITY'],gpt2_xl_default['PROFANITY'],llama_default['PROFANITY']]
severe_toxicity_1 = [gpt2_default['SEVERE_TOXICITY'],gpt2_medium_default['SEVERE_TOXICITY'],gpt2_large_default['SEVERE_TOXICITY'],gpt2_xl_default['SEVERE_TOXICITY'],llama_default['SEVERE_TOXICITY']]
insult_1 = [gpt2_default['INSULT'],gpt2_medium_default['INSULT'],gpt2_large_default['INSULT'],gpt2_xl_default['INSULT'],llama_default['INSULT']]
threat_1 = [gpt2_default['THREAT'],gpt2_medium_default['THREAT'],gpt2_large_default['THREAT'],gpt2_xl_default['THREAT'],llama_default['THREAT']]
toxicity_1 = [gpt2_default['TOXICITY'],gpt2_medium_default['TOXICITY'],gpt2_large_default['TOXICITY'],gpt2_xl_default['TOXICITY'],llama_default['TOXICITY']]


attributes_1 = [identity_attack_1, profanity_1, severe_toxicity_1, insult_1, threat_1, toxicity_1]
avg_1 = [0,0,0,0,0]

for i in range(len(attributes_1)):
    # plt.scatter(model, attributes_1[i], label=labels[i])
    for j in range(5):
        avg_1[j]+=attributes_1[i][j]
for j in range(5):
    avg_1[j]/=6

# plt.plot(model, avg_1, color='blue')


### debiasing graph

gpt2_debiased = {'THREAT': 0.03857057849460616, 'TOXICITY': 0.24479542517885286, 'SEVERE_TOXICITY': 0.059297423830051474, 'INSULT': 0.12736493819409223, 'PROFANITY': 0.19292272228253424, 'IDENTITY_ATTACK': 0.04338121797099314}
gpt2_medium_debiased = {'IDENTITY_ATTACK': 0.05349331767638066, 'TOXICITY': 0.29602975860016945, 'INSULT': 0.16399588206652482, 'THREAT': 0.04444914189413768, 'PROFANITY': 0.2346819021610024, 'SEVERE_TOXICITY': 0.07408845069551406}
gpt2_large_debiased = {'TOXICITY': 0.2893333544933672, 'IDENTITY_ATTACK': 0.051928628608171865, 'PROFANITY': 0.23013353832568023, 'THREAT': 0.04459566434396266, 'INSULT': 0.15760282383886037, 'SEVERE_TOXICITY': 0.07080134114652217}
gpt2_xl_debiased = {'PROFANITY': 0.2218957125199827, 'TOXICITY': 0.28393420220365634, 'SEVERE_TOXICITY': 0.06462540007106303, 'INSULT': 0.1518972512512751, 'THREAT': 0.045534652117346974, 'IDENTITY_ATTACK': 0.05328601964141167}
llama_debiased = {'THREAT': 0.046794640581810584, 'SEVERE_TOXICITY': 0.05974380558093047, 'IDENTITY_ATTACK': 0.06040582507524734, 'INSULT': 0.1625439865500417, 'TOXICITY': 0.2937590617409051, 'PROFANITY': 0.2182627748977369}

identity_attack_2 = [gpt2_debiased['IDENTITY_ATTACK'],gpt2_medium_debiased['IDENTITY_ATTACK'],gpt2_large_debiased['IDENTITY_ATTACK'],gpt2_xl_debiased['IDENTITY_ATTACK'],llama_debiased['IDENTITY_ATTACK']]
profanity_2 = [gpt2_debiased['PROFANITY'],gpt2_medium_debiased['PROFANITY'],gpt2_large_debiased['PROFANITY'],gpt2_xl_debiased['PROFANITY'],llama_debiased['PROFANITY']]
severe_toxicity_2 = [gpt2_debiased['SEVERE_TOXICITY'],gpt2_medium_debiased['SEVERE_TOXICITY'],gpt2_large_debiased['SEVERE_TOXICITY'],gpt2_xl_debiased['SEVERE_TOXICITY'],llama_debiased['SEVERE_TOXICITY']]
insult_2 = [gpt2_debiased['INSULT'],gpt2_medium_debiased['INSULT'],gpt2_large_debiased['INSULT'],gpt2_xl_debiased['INSULT'],llama_debiased['INSULT']]
threat_2 = [gpt2_debiased['THREAT'],gpt2_medium_debiased['THREAT'],gpt2_large_debiased['THREAT'],gpt2_xl_debiased['THREAT'],llama_debiased['THREAT']]
toxicity_2 = [gpt2_debiased['TOXICITY'],gpt2_medium_debiased['TOXICITY'],gpt2_large_debiased['TOXICITY'],gpt2_xl_debiased['TOXICITY'],llama_debiased['TOXICITY']]


attributes_2 = [identity_attack_2, profanity_2, severe_toxicity_2, insult_2, threat_2, toxicity_2]
avg_2 = [0,0,0,0,0]

for i in range(len(attributes_2)):
    # plt.scatter(model, attributes_2[i], label=labels[i])
    for j in range(5):
        avg_2[j]+=attributes_2[i][j]
for j in range(5):
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
plt.xlabel('Model', fontweight='bold')
plt.ylabel('Average probabilities', fontweight='bold')
# plt.legend(fontsize=8)

idx = np.arange(5)

w = 0.3

plt.bar(idx-0.5*w, avg_1, label='default', width=w, color='#FFB6C1')
plt.bar(idx+0.5*w, avg_2, label='debiased', width=w, color='#B0E0E6')
plt.xticks(idx, model)

plt.legend(fontsize=8, ncol=1)



plt.savefig('image_debiasing_rate.png',bbox_inches='tight')

