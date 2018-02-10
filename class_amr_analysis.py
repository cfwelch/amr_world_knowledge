

def main():
    train_aligns = open("train.txt.amr.tok.aligned");
    train_lines = train_aligns.readlines();
    fives = 0;
    n_fives = 0;
    class_dict = {};
    for line in train_lines:
        if line.startswith("# ::node"):
            nodeline = line.strip().split();
            if len(nodeline) != 5:
                n_fives += 1;
            else:
                fives += 1;
                if nodeline[3] in class_dict:
                    class_dict[nodeline[3]] += 1;
                else:
                    class_dict[nodeline[3]] = 1;
    print("fives: " + str(fives));
    print("n_fives: " + str(n_fives));
    print(class_dict);


if __name__ == "__main__":
    main();
