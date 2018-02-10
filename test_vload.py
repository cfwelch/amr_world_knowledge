import numpy as np

def get_embeddings(emb_binary_file=None):
	emb_dictionary = None
	if emb_binary_file == None:
		return emb_dictionary
	with open(emb_binary_file, "rb") as f:
		header = f.readline()
		emb_dictionary = {}
		num_embeddings, emb_size = map(int, header.split())
		binary_len = np.dtype('float32').itemsize * emb_size
		for line in xrange(num_embeddings):
			word = []
			while True:
				ch = f.read(1)
				if ch == ' ':
					word = ''.join(word)
					break
				if ch != '\n':
					word.append(ch)
			emb_dictionary[word] = np.fromstring(f.read(binary_len), dtype='float32')
		return  emb_dictionary

z = get_embeddings("vectors.bin")

title_list = []
title_file = open("titles-sorted.txt")
title_lines = title_file.readlines()
title_file.close()
for line in title_lines:
    title_list.append(line.strip().replace("_", " "))

name = "German Flag".replace("_", " ")

tindex = title_list.index(name)+1
print("index is: " + str(tindex))

a = z[str(tindex)]

name = "Germany".replace("_", " ")
tindex = title_list.index(name)+1
print("index is: " + str(tindex))

b = z[str(tindex)]


dist = numpy.linalg.norm(a-b)
print("distance is: " + str(dist))