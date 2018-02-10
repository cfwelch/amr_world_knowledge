
import nltk, sys, copy
from nltk.corpus import wordnet as wn

def main():
    if len(sys.argv) > 1:
        inword = sys.argv[1];
        print("Finding synsets for " + str(inword));
        synsets = wn.synsets(inword, lang='eng');
        oset = list();
        finals = list();
        for synset in synsets:
            oset.append([synset]);
        while len(oset) > 0:
            last = oset.pop();
            hypers = last[-1:][0].hypernyms();
            if len(hypers) > 0:
                for hyper in hypers:
                    temp = last[:];
                    temp.append(hyper);
                    oset.append(temp);
            else:
                finals.append(last);
        print("================ Final Traces ================");
        final_abstracts = set();
        for entry in finals:
            print(entry);
            if entry[0]._pos == "v":
                if len(entry) > 2:
                    final_abstracts.add(entry[len(entry)-3]._name);
            else:
                if len(entry) > 4:
                    final_abstracts.add(entry[len(entry)-5]._name);
        print("================ Abstract Layer ================");
        print(final_abstracts);
    else:
        print("Not enough command line arguments...");

def getSupersenses(word):
    synsets = wn.synsets(word, lang='eng');
    oset = set();
    # to use all synset names
    #for synset in synsets:
    #    oset.add(synset.lexname());
    if len(synsets) > 0:
        oset.add(synsets[0].lexname());
    return oset;

def getConcepts(word):
    synsets = wn.synsets(word, lang='eng');
    oset = list();
    finals = list();
    for synset in synsets:
        oset.append([synset]);
    while len(oset) > 0:
        last = oset.pop();
        hypers = last[-1:][0].hypernyms();
        if len(hypers) > 0:
            for hyper in hypers:
                temp = last[:];
                temp.append(hyper);
                oset.append(temp);
        else:
            finals.append(last);
    final_abstracts = set();
    for entry in finals:
        if entry[0]._pos == "v":
            pass;
            #final_abstracts.add(entry[len(entry)-1]._name);
        else:
            if len(entry) > 2:
                final_abstracts.add(entry[len(entry)-3]._name);
    return final_abstracts;

if __name__ == "__main__":
    main();
