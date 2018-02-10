

def main():
    train_aligns = open("train.txt.amr.tok.aligned");
    train_lines = train_aligns.readlines();
    label_sets = {"KEY_ERROR": 0};
    last_tokens = [];
    last_node_map = {};
    last_node_indicies = [];
    line_index = 0;
    for line in train_lines:
        if line.startswith("# ::tok"):
            if last_tokens != []:
                for last_index in last_node_indicies:
                    if last_index[:-2] in last_node_map:
                        index_key = last_node_map[last_index[:-2]];
                        if index_key in label_sets:
                            label_sets[index_key] += 1;
                        else:
                            label_sets[index_key] = 1;
                    else:
                        label_sets["KEY_ERROR"] += 1;
                    #print(last_node_map[last_index[:-2]]);
                last_node_map = {};
                last_node_indicies = [];
                line_index += 1;
            last_tokens = line[8:].split();
        elif line.startswith("# ::node"):
            nodeline = line.strip().split();
            if len(nodeline) == 5:
                last_node_map[nodeline[2]] = nodeline[3];
                if nodeline[3] == "name":
                    last_node_indicies.append(nodeline[2]);
    print(label_sets);

if __name__ == "__main__":
    main();
