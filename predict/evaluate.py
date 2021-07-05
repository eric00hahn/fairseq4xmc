import re
import matplotlib.pyplot as plt
import random

def compute_f1() -> None :
    f = open('./amazoncat-13k/tst_fconv_self_att.out', 'r')
    lines = f.readlines()
    N_max = int( len(lines) / 5)
    precision_cumulated = 0
    recall_cumulated = 0
    jaccard_sim_cumulated = 0
    for N in range(N_max) :        
        t = 5 * N
        
        index = lines[t]
        index = index.split('\t2', 1)[0]
        index = int( re.sub('[^0-9]','', index) ) + 1

        target = lines[t+1]
        target = target.split('\t', 1)[-1].rstrip()
        target = target.replace(" &amp; ", "&")
        target = target.replace(" &apos;s", "'s")
        target_set = set( target.split(' ') )
        
        predicted = lines[t+2]
        predicted = predicted.split('\t', 1)[-1]
        predicted = predicted.split('\t', 1)[-1].rstrip()    
        predicted = predicted.replace(" &amp; ", "&")
        predicted_set = set( predicted.split(' ') )
        
        intersection = float(len(target_set.intersection(predicted_set)))
        union = float(len(target_set.union(predicted_set)))

        precision_cumulated += intersection / float(len(predicted_set))
        recall_cumulated += intersection / float(len(target_set))
        jaccard_sim_cumulated += intersection / union
    
        print("prec :", intersection / float(len(predicted_set)))
        print("reca :", intersection / float(len(target_set)))
        print("~~~~~~~~~~~~~~~~")

    precision = precision_cumulated / N_max
    recall = recall_cumulated / N_max
    f1 = ( 2 * precision * recall ) / (precision + recall)
    j = jaccard_sim_cumulated / N_max

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Macro-F1 :", f1)
    print("Jac :", j)
    return None

def compute_Pk(k : int) -> None :
    f = open('./amazoncat-13k/tst_fconv_self_att.out', 'r')
    lines = f.readlines()
    N_max = int( len(lines) / 5)
    p_at_k_cumulated = 0
    for N in range(N_max) :        
        t = 5 * N
        index = lines[t]
        index = index.split('\t2', 1)[0]
        index = int( re.sub('[^0-9]','', index) ) + 1

        target = lines[t+1]
        target = target.split('\t', 1)[-1].rstrip()
        target = target.replace(" &amp; ", "&")
        target = target.replace(" &apos;s", "'s")
        target = target.split(' ') 
        target = list(set(target))
        target.sort()

        predicted = lines[t+2]
        predicted = predicted.split('\t', 1)[-1]
        predicted = predicted.split('\t', 1)[-1].rstrip()    
        predicted = predicted.replace(" &amp; ", "&")
        predicted = predicted.split(' ')
        predicted = list(set(predicted))
        predicted.sort()
        predicted += 100 * ['EMPTY_LABEL']

        top_k = [predicted[i] for i in range(k)]
        # top_k = random.sample(predicted, k)

        p_at_k = float(len(set(target).intersection(set(top_k)))) / k
        p_at_k_cumulated += p_at_k 
        
    print("P@" + str(k) + ":", p_at_k_cumulated / N_max)
    return p_at_k_cumulated / N_max

def plot_precision_at_k() :
    x = []
    y = []
    for k in range(1,6) :
        x += [k]
        y += [compute_Pk(k)]
    plt.style.use('seaborn-whitegrid')
    plt.plot(x, 
             y, 
             linestyle='--',
             linewidth=0.5,
             color='black',
             marker='x',
             label='seq2seq')
    plt.plot([1,3,5], 
             [0.9592, 0.8241, 0.6731],
             linestyle='--',
             linewidth=0.5,
             color='b',
             marker='x',
             label='AttentionXML')
    plt.title("P@k on AmazonCat-13K")
    plt.xlabel("k")
    plt.ylabel("P@k")
    plt.legend(loc="upper right")
    plt.savefig("./plots/amazoncat13k_pk.png", dpi=300)
    return None

def main() -> int :
   
    plot_precision_at_k()
    plt.show()

    return 0

if __name__ == '__main__' :
    main()
