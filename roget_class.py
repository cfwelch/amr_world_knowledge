
from os import listdir
from os.path import isfile, join
import sys

cat_list = dict();

def onImport():
	mypath = "../../data/roget_processed";
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))];
	for f_name in onlyfiles:
		f_temp = open(mypath + "/" + f_name);
		f_lines = f_temp.readlines();
		for line in f_lines:
			parts = line.strip().split(",");
			if parts[0] in cat_list:
				cat_list[parts[0]].add(parts[1]);
			else:
				cat_list[parts[0]] = {parts[1]};

def getClass(in_str):
	if in_str in cat_list:
		return list(cat_list[in_str]);
	else:
		return [];

#print(cat_list);

#print("The categories of \"" + sys.argv[1] + "\" are: " + str(cat_list[sys.argv[1]]));

if __name__ != "__main__":
	onImport();

