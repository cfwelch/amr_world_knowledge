
#AFTER DEV CUT
classes = ['country', 'person', 'publication', 'continent', 'military', 'religious-group', 'province'];
#classes = ['country', 'person', 'organization', 'city', 'government-organization', 'company', 'world-region', 'publication', 'continent', 'political-party', 'criminal-organization', 'military', 'religious-group', 'province', 'treaty', 'newspaper', 'ethnic-group', 'product', 'state', 'country-region'];
#, 'aircraft-type', 'group', 'facility', 'research-institute', 'thing', 'event', 'war', 'law', 'agency', 'university'

#AFTER DEV CUT
class_abbr = ["CO", "PE", "PU", "CN", "MI", "RG", "PR"];
#class_abbr = ["CO", "PE", "OR", "CI", "GO", "CM", "WR", "PU", "CN", "PP", "CR", "MI", "RG", "PR", "TR", "NE", "EG", "PD", "ST", "CU"];
#"AT", "GR", "FA", "RI", "TH", "EV", "WA", "LA", "AG", "UN"];

def main():
    print("Length of classes: " + str(len(classes)));
    print("Length of class_abbr: " + str(len(class_abbr)));
    #######################################################
    train_prp = open("./for_gold_amrner_gen_only/get_gold.txt.sent.prp");
    #train_prp = open("train.txt.sent.prp");
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
    last_node_map = {};
    last_node_indicies = [];
    last_node_index_spans = [];
    line_index = 0;
    for line in train_lines:
        if line.startswith("# ::tok"):
            if last_tokens != []:
    ######################################################### INDEX LOOP #########################################################
                for last_index in range(0, len(last_node_indicies)):
                    if last_node_indicies[last_index][:-2] in last_node_map:
                        index_key = last_node_map[last_node_indicies[last_index][:-2]];
                        if index_key in classes:
                            t_range = last_node_index_spans[last_index];
                            for i in range(t_range[0], t_range[1]):
                                prefix = "I";
                                if i == t_range[0]:
                                    prefix = "B";
                                cur_label_set[i] = prefix + "-" + class_abbr[classes.index(index_key)];
    ######################################################### INDEX LOOP #########################################################
                last_node_map = {};
                last_node_indicies = [];
                last_node_index_spans = [];
                label_sets.append(cur_label_set);
                line_index += 1;
            last_tokens = line[8:].split();
            cur_label_set = ['O'] * len(token_sets[line_index]);
        elif line.startswith("# ::node"):
            nodeline = line.strip().split();
            if len(nodeline) == 5:
                last_node_map[nodeline[2]] = nodeline[3];
                if nodeline[3] == "name":
                    last_node_indicies.append(nodeline[2]);
                    _ndline = nodeline[4].split("-");
                    start = int(_ndline[0]);
                    end = int(_ndline[1]);
                    last_node_index_spans.append((start,end));
    ######################################################### INDEX LOOP #########################################################
    for last_index in range(0, len(last_node_indicies)):
        if last_node_indicies[last_index][:-2] in last_node_map:
            index_key = last_node_map[last_node_indicies[last_index][:-2]];
            if index_key in classes:
                t_range = last_node_index_spans[last_index];
                for i in range(t_range[0], t_range[1]):
                    prefix = "I";
                    if i == t_range[0]:
                        prefix = "B";
                    cur_label_set[i] = prefix + "-" + class_abbr[classes.index(index_key)];
    ######################################################### INDEX LOOP #########################################################
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
