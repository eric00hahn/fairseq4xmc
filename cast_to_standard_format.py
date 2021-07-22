import json
import pprint as pp

def cast_amazoncat13k(trn_or_tst : str) -> None :
    f = open("./data/amazoncat13k/raw/" + trn_or_tst + ".json")
    lines = f.readlines()
    g = open("./data/amazoncat13k/bow/Yf_utf8.txt")
    lines_labels = g.readlines()

    print("Computing all labels... ")
    labels = []
    for line in lines :
        labels += json.loads(line)['target_ind']
    labels = list(set(labels))
    
    label_frequencies = {}
    label_explanations = {}
    for l in labels :
        label_frequencies[l] = 0
        label_explanations[l] = lines_labels[l]
    print("Computing absolute label frequencies...")
    for line in lines :
        current_labels = json.loads(line)['target_ind']
        for label in current_labels :
            label_frequencies[label] += 1

    d = {}
    line_counter = 0
    for line in lines :
        x = json.loads(line)
        d[str(line_counter)] = {}
        d[str(line_counter)]['document'] = x['title'] + " " + x['content']
        d[str(line_counter)]['labels'] = x['target_ind']
        d[str(line_counter)]['label_frequencies_absolute'] = {}
        d[str(line_counter)]['label_frequencies_relative'] = {}
        d[str(line_counter)]['label_explanations'] = {}
        for l in x['target_ind'] :
            d[str(line_counter)]['label_frequencies_absolute'][l] = label_frequencies[l]
            d[str(line_counter)]['label_frequencies_relative'][l] = label_frequencies[l] / len(lines)
            d[str(line_counter)]['label_explanations'][l] = label_explanations[l].replace('\n', '')
        if line_counter % 100000 == 0 :
            print("Processed :", line_counter)
        line_counter += 1 
    d['all_base_label_frequencies'] = label_frequencies
    d['all_base_label_explanations'] = label_explanations
    with open("./data/amazoncat13k/standard/" + trn_or_tst + ".json", "w") as outfile : 
        json.dump(d, outfile)
    return None

def cast_amazontitles670k(trn_or_tst : str) -> None :
    f = open("./data/amazontitles670k/raw/" + trn_or_tst + ".json")
    lines = f.readlines()

    print("Computing all labels... ")
    labels = []
    for line in lines :
        labels += json.loads(line)['target_ind']
    labels = list(set(labels))

    label_frequencies = {}
    label_explanations = {}
    for l in labels :
        label_frequencies[l] = 0
        label_explanations[l] = ''
    print("Computing absolute label frequencies...")
    for line in lines :
        current_labels = json.loads(line)['target_ind']
        for label in current_labels :
            label_frequencies[label] += 1

    d = {}
    line_counter = 0
    for line in lines :
        x = json.loads(line)
        d[str(line_counter)] = {}
        d[str(line_counter)]['document'] = x['title'] + " " + x['content']
        d[str(line_counter)]['labels'] = x['target_ind']
        
        d[str(line_counter)]['label_frequencies_absolute'] = {}
        d[str(line_counter)]['label_frequencies_relative'] = {}
        d[str(line_counter)]['label_explanations'] = {}
        for l in x['target_ind'] :
            d[str(line_counter)]['label_frequencies_absolute'][l] = label_frequencies[l]
            d[str(line_counter)]['label_frequencies_relative'][l] = label_frequencies[l] / len(lines)
            d[str(line_counter)]['label_explanations'][l] = ''
        if line_counter % 100000 == 0 :
            print("Processed :", line_counter)
        line_counter += 1 
    d['all_base_label_frequencies'] = label_frequencies
    d['all_base_label_explanations'] = label_explanations
    with open("./data/amazontitles670k/standard/" + trn_or_tst + ".json", "w") as outfile : 
        json.dump(d, outfile)
    return None

def cast_lfamazontitle130k(trn_or_tst : str) -> None :
    f = open("./data/lfamazon131k/raw/" + trn_or_tst + ".json")
    lines = f.readlines()
    g = open("./data/lfamazon131k/bow/Yf.txt")
    lines_labels = g.readlines()

    print("Computing all labels... ")
    labels = []
    for line in lines :
        labels += json.loads(line)['target_ind']
    labels = list(set(labels))
 
    return None

def main() -> int :

    cast_lfamazontitle130k('trn')


    return 0

if __name__ == '__main__' :
    main()
