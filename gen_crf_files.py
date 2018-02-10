
#i think this is what i did first and it did the best... minus state-01
classes = ['name', 'country', 'person', 'date-entity', 'government-organization', 'organization', 'thing', 'city', 'company', 'temporal-quantity', 'military'];
class_abbr = ['NA', 'CO', 'PE', 'DE', 'GO', 'OR', 'TH', 'CI', 'CM', 'TQ', 'MI'];


#################################These classes are for set of things which have 'name' children
#AFTER DEV CUT                       -------- how did i get 0.66.... i'm removing name again...
#classes = ['country', 'government-organization', 'publication', 'continent', 'military', 'religious-group', 'province', 'treaty', 'newspaper'];
#'person', 'organization', 'city', 'company', 'world-region', 'political-party', 'criminal-organization', 'ethnic-group', 'product', 'state', 'country-region'
#AFTER DEV CUT
#class_abbr = ["CO", "GO", "PU", "CN", "MI", "RG", "PR", "TR", "NE"];
#"PE", "OR", "CI", "CM", "WR", "PP", "CR", "EG", "PD", "ST", "CU"
#################################################################################################

def main():
    print("Length of classes: " + str(len(classes)));
    print("Length of class_abbr: " + str(len(class_abbr)));
    #######################################################
    #train_prp = open("train.txt.sent.prp");
    train_prp = open("./for_gold_amrner_gen_only/get_gold.txt.sent.prp");
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
    train_aligns = open("./for_gold_amrner_gen_only/get_gold.txt.amr.tok.aligned");
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
            if len(nodeline) == 5:
                if nodeline[3] in classes:
                    #print(nodeline[3] + "---" + nodeline[4]);
                    _ndline = nodeline[4].split("-");
                    start = int(_ndline[0]);
                    end = int(_ndline[1]);
                    for i in range(start, end):
                        prefix = "I";
                        if i == start:
                            prefix = "B";
                        #print(cur_label_set);
                        cur_label_set[i] = prefix + "-" + class_abbr[classes.index(nodeline[3])];
    label_sets.append(cur_label_set);
    #######################################################
    out_file = open("CRF-AMR-NER-test", "w");
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



"""OLD classes#classes = ['name', 'country', 'and', 'person', 'have-org-role-91', 'date-entity', 'state-01', 'government-organization', 'organization', 'thing', 'city', 'possible', 'govern-01', 'this', 'say-01', 'nucleus', 'company', 'i', 'temporal-quantity', 'international', 'military', 'include-91', 'drug', 'weapon', 'year', 'official'];
#classes = ['name', 'country', 'and', 'date-entity', 'state-01', 'organization', 'city', 'possible', 'govern-01', 'this', 'say-01', 'nucleus', 'company', 'i', 'temporal-quantity', 'international', 'military', 'drug', 'weapon', 'year', 'official'];
#classes = ['name', 'country', 'person', 'date-entity', 'state-01', 'government-organization', 'organization', 'thing', 'city', 'company', 'temporal-quantity', 'military'];
#class_abbr = ['NA', 'CO', 'PE', 'DE', 'ST', 'GO', 'OR', 'TH', 'CI', 'CM', 'TQ', 'MI'];
class_abbr = ['NA', 'CO', 'AN', 'DE', 'ST', 'OR', 'CI', 'PO', 'GN', 'TI', 'SA', 'NU', 'CM', 'II', 'TQ', 'IN', 'MI', 'DR', 'WE', 'YR', 'OF'];"""
