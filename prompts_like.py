import glob



files = glob.glob('./debiasing_output/prompted_generations_gpt2_debiased.txt')

with open('./collect.txt','w') as wf:

    with open('./debiasing_output/prompted_generations_gpt2_debiased.txt','r') as rf:
        cnt = 0
        for line in rf.readlines():
            cnt+=1
            ps = line.find('"prompt": ') + 10
            pf = line.find(', "challenging"')
            print(line[ps:pf])
            wf.write(line[ps:pf] + '\n')