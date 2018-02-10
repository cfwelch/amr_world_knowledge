
import cPickle as pickle

def main():
    #f_ = open("CRF-AMR-NER-dev");
    #f_ = open("CRF-AMR-NER-test");
    #f_ = open("CRF-AMR-NER-train");
    f_ = open("labeled_out_orig");
    f_lines = f_.readlines();
    o_ = open("test-sentences.txt.amrner", "wb");
    #o_ = open("train.txt.amrner", "wb");
    #o_ = open("dev.txt.amrner", "wb");
    arr = [];
    larr = [];
    for line in f_lines:
        if line.strip() == "":
            if larr != []:
                arr.append(larr);
                larr = [];
        else:
            tline = line.strip().split("\t");
            larr.append(tline[2]);

    pickle.dump(arr, o_);

if __name__ == "__main__":
    main();
