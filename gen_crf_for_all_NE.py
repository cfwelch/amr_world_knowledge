
all_nes = {'cleric', 'shop', 'cosmodrome', 'agency', 'bomb', 'spaceship', 'rocket', 'show', 'speech', 'mission', 'dynasty', 'medicine', 'radar', 'radio', 'gang', 'victim', 'range', 'KEY_ERROR', 'continent', 'reactor', 'site', 'socialite', 'group', 'destroyer', 'committee', 'complex', 'platform', 'program', 'location', 'policy', 'political-party', 'holiday', 'pill', 'town', 'myth', 'clear-01', 'stop', 'nation', 'band', 'game', 'report', 'township', 'bank', 'school', 'prize', 'university', 'religious-group', 'brother', 'bay', 'rifle', 'subway', 'team', 'quote-01', 'river', 'page', 'husband', 'country-region', 'attorney', 'ethnic-group', 'change-01', 'series', 'village', 'satellite', 'module', 'street', 'draft', 'plant', 'sea', 'pass', 'sports-facility', 'magazine', 'operation', 'cartel', 'event', 'accomplice', 'consortium', 'project', 'canvas', 'call-01', 'revolution', 'network', 'movie', 'identify-01', 'milk', 'state', 'activist', 'worship-place', 'local-region', 'capital', 'name-01', 'princess', 'temple', 'movement', 'body', 'sheep', 'strain', 'theory', 'exhibit-01', 'journal', 'government-organization', 'incident', 'base', 'address', 'natural-disaster', 'earthquake', 'vaccine', 'publication', 'league', 'comrade', 'daughter', 'jet', 'language', 'country', 'region', 'drug', 'engine', 'thing', 'airport', 'period', 'noodle', 'use-01', 'castle', 'documentary', 'museum', 'road', 'software', 'alliance', 'family', 'facility', 'battery', 'date-entity', 'mention-01', 'world-region', 'businessman', 'son', 'county', 'kill-01', 'initiative', 'malfunction', 'spell-01', 'car-make', 'watchdog', 'conference', 'city', 'aircraft-type', 'district', 'area', 'festival', 'hospital', 'airliner', 'system', 'legend', 'submarine', 'station', 'music', 'card', 'lot', 'vehicle', 'helicopter', 'territory', 'war', 'website', 'boy', 'club', 'company', 'hotel', 'park', 'sister', 'award', 'missile', 'peninsula', 'train', 'broadcast-program', 'subsidiary', 'institution', 'case', 'television', 'valley', 'politician', 'monopoly', 'title-01', 'worm', 'establish-01', 'planet', 'island', 'newspaper', 'and', 'firm', 'lawyer', 'palace', 'strait', 'criminal-organization', 'bear-01', 'dissident', 'dispersant', 'laboratory', 'ship', 'treaty', 'seminar', 'regime', 'mountain', 'doctor', 'idiot', 'corporation', 'research-institute', 'brand', 'tour', 'port', 'field', 'book', 'republic', 'animal', 'enclave', 'department', 'supertanker', 'document', 'analyze-01', 'channel', 'province', 'stone', 'product', 'benchmark', 'monastery', 'class', 'mother', 'meet-03', 'suburb', 'virus', 'plane', 'statue', 'military', 'city-district', 'law', 'desert', 'man', 'building', 'journalist', 'cosmonaut', 'center', 'wife', 'camp', 'disease', 'ocean', 'person', 'nickname-01', 'know-01', 'multivitamin', 'organization', 'model', 'homophone', 'moon', 'railway-line'};

def main():
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
            #print(nodeline);
            if len(nodeline) == 5:
                if "Republican Party" not in line and "0.1.0.0.0.1.0.0.1.0.0.0" not in line:
                    if nodeline[3] not in all_nes:###########add a not in here if you want to separate the nes from the non-nes
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
