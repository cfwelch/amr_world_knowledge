

#classes = ['name', 'country', 'and', 'person', 'have-org-role-91', 'date-entity', '-', 'state-01', 'government-organization', 'organization', 'thing', 'city', 'possible', 'govern-01', 'this', 'say-01'];
#classes = ['name', 'country', 'person', 'date-entity', 'state-01', 'government-organization', 'organization', 'thing', 'city', 'company', 'temporal-quantity', 'military'];

# expanded classes
classes = ['name', 'country', 'and', 'person', 'have-org-role-91', 'date-entity', 'state-01', 'government-organization', 'organization', 'thing', 'city', 'possible', 'govern-01', 'this', 'say-01', 'nucleus', 'company', 'i', 'temporal-quantity', 'international', 'military', 'include-91', 'drug', 'weapon', 'year', 'official'];

def main():
    train_aligns = open("train.txt.amr.tok.aligned");
    train_lines = train_aligns.readlines();
    fives = 0;
    legits = 0;
    n_fives = 0;
    class_dict = {};
    last_tokens = [];
    for line in train_lines:
        if line.startswith("# ::tok"):
            last_tokens = line[8:].split();
        elif line.startswith("# ::node"):
            nodeline = line.strip().split();
            if len(nodeline) != 5:
                n_fives += 1;
            else:
                fives += 1;
                if nodeline[3] in classes:
                    legits += 1;
                #print(last_tokens);
                #print(nodeline);
                if nodeline[3] in class_dict:
                    class_dict[nodeline[3]] += 1;
                else:
                    class_dict[nodeline[3]] = 1;
    #print(class_dict);
    print("fives: " + str(fives));
    print("legits: " + str(legits));


if __name__ == "__main__":
    main();
