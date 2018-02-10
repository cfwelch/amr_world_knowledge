

def main():
    #######################################################
    train_prp = open("dev.txt.sent.prp");
    #train_prp = open("./for_gold_amrner_gen_only/get_gold.txt.sent.prp");
    train_plines = train_prp.readlines();
    text_lines = [];
    token_sets = [];
    text_sets = [];
    for line in train_plines:
        if line.startswith("[Text"):
            lparts = line.strip()[1:-1].split("] [");
            text_lines.append(lparts);
            token_set = [];
            text_set = [];
            for lpart in lparts:
                tparts = lpart.split(" ");
                for tpart in tparts:
                    if tpart.startswith("PartOfSpeech"):
                        token_set.append(tpart[13:]);
                    elif tpart.startswith("Text"):
                        text_set.append(tpart[5:]);
            token_sets.append(token_set);
            text_sets.append(text_set);
    print("Length of POS sets: " + str(len(token_sets)));
    print("Length of text lines: " + str(len(text_lines)));
    #print(token_sets[10311]);
    #######################################################
    train_aligns = open("dev.txt.amr.tok.aligned");
    train_lines = train_aligns.readlines();
    label_sets = [];
    cur_label_set = [];
    last_tokens = [];
    line_index = 0;
    for line in train_lines:
        if line.startswith("# ::tok"):
            if last_tokens != []:
                label_sets.append(cur_label_set);
                line_index += 1;
            last_tokens = line[8:].split();
            cur_label_set = ['O'] * len(token_sets[line_index]);
        elif line.startswith("# ::node"):
            nodeline = line.strip().split();
            #print(nodeline);
            if len(nodeline) == 5:
                if "Republican Party" not in line and "0.1.0.0.0.1.0.0.1.0.0.0" not in line:
                    #print(nodeline[3] + "---" + nodeline[4]);
                    _ndline = nodeline[4].split("-");
                    start = int(_ndline[0]);
                    end = int(_ndline[1]);
                    for i in range(start, end):
                        prefix = "I";
                        if i == start:
                            prefix = "B";
                        #print(cur_label_set);
                        cur_label_set[i] = prefix + "-" + nodeline[3].upper().replace("_", "0").replace("-", "0").replace("\"", "").replace("'", "");
    label_sets.append(cur_label_set);
    #######################################################
    out_file = open("CRF-AMR-NER-dev", "w");
    for tset in range(0, len(text_sets)):
        for _i in range(0, len(text_sets[tset])):
            #print(str(len(text_sets[tset])) + str(len(token_sets[tset])) + str(len(label_sets[tset])));
            #print(text_sets[tset]);
            #print(token_sets[tset]);
            if _i < len(token_sets[tset]):
                out_file.write(text_sets[tset][_i] + "\t" + token_sets[tset][_i] + "\t" + label_sets[tset][_i] + "\n");
        out_file.write("\n");
    #######################################################

if __name__ == "__main__":
    main();
