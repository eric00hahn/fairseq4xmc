import json
import time

def read_source() -> None :
    path_in = "/home/erichahn/src/ma/data/Amazon-Cat-13K/tst.json"
    path_out = "/home/erichahn/src/ma/src/fairseq-amazoncat-13k/data/tst.src"

    with open(path_in) as in_file :
        lines = in_file.readlines()

    """ delete file contents of out_file on call of read_documents() """
    # open(path_out, 'w').close()

    print("-------Reading from :" + path_in)
    print("-------Writing to :" + path_out)
    print("-------Total line count : ", len(lines))

    start_time = time.time()
    line_counter = 0
    with open(path_out, 'a') as out_file :
        for line in lines :
            if line_counter % 500 == 0 :
                print("Processed lines :", line_counter, "--- %s minutes ---" % ( (time.time() - start_time) / 60 ))
            entry = json.loads(line)
            document = entry['content']
            out_file.write(document)
            out_file.write('\n')
            line_counter += 1
    return None

def read_target() -> None :
    path_labels = "/home/erichahn/src/ma/data/Amazon-Cat-13K/Yf.txt"
    with open(path_labels) as labels_file :
        lines = labels_file.readlines()
    index2label = {}
    line_counter = 0
    for line in lines :
        # line = line.replace(" ", "")
        line = line.rstrip()
        index2label[line_counter] = line
        line_counter += 1

    path_in = "/home/erichahn/src/ma/data/Amazon-Cat-13K/trn.json"
    path_out = "/home/erichahn/src/ma/src/fairseq-amazoncat-13k/data/trn.tgt"
    
    """ delete file contents of out_file on call of read_documents() """
    open(path_out, 'w').close()

    print("reading from ..." + path_in)
    print("total line count : ", len(lines))

    with open(path_in) as in_file :
        lines = in_file.readlines()
    
    start_time = time.time()
    line_counter = 0
    with open(path_out, 'a') as out_file :
        for line in lines :
            if line_counter % 500 == 0 :
                print("Processed lines :", line_counter, "--- %s minutes ---" % ( (time.time() - start_time) / 60 ))
            entry = json.loads(line)
            target_labels = entry['target_ind']
            for t in target_labels :
                out_file.write(index2label[t])
                out_file.write(' ')
            out_file.write('\n')
            line_counter += 1
    return None

def main() -> int :

    read_target()

    return 0

if __name__ == '__main__' :
    main()