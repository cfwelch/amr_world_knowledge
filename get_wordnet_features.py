
import wordnet_features
import cPickle as pickle

sentences = open("test-sentences.txt.tok");
out = open("test-sentences.txt.wordnet_features", "wb");
#sentences = open("train.txt.sent.tok");
#out = open("train.txt.wordnet_supersenses", "wb");
#sentences = open("dev.txt.sent.tok");
#out = open("dev.txt.wordnet_supersenses", "wb");
sents = sentences.readlines();

count = 0;
output = list();
for i in sents:
	_i = i.decode('utf-8').strip();
	parts = _i.split();
	t_dict = dict();
	line = list();
	for part in parts:
		token = part.lower();
		if token.endswith(";") or token.endswith("."):
			token = token[:-1];
		rcs = wordnet_features.getConcepts(token);#getSupersenses 
		line.append(rcs);
	#print(line);
	output.append(line);
	count += 1;
	print(count*1.0/len(sents));

pickle.dump(output, out);
