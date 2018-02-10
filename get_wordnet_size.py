
import cPickle as pickle

fullset = set();
test_file = open("test-sentences.txt.wordnet_features", "rb");#wordnet_supersenses
train_file = open("train.txt.wordnet_features", "rb");#wordnet_supersenses
dev_file = open("dev.txt.wordnet_features", "rb");#wordnet_supersenses

test_set = pickle.load(test_file);
train_set = pickle.load(train_file);
dev_set = pickle.load(dev_file);

for i in test_set:
    for ii in i:
        for iii in ii:
            fullset.add(iii);

for i in train_set:
    for ii in i:
        for iii in ii:
            fullset.add(iii);

for i in dev_set:
    for ii in i:
        for iii in ii:
            fullset.add(iii);

print("The size of the full set is " + str(len(fullset)));
print(fullset);
