import urllib2
import urllib

sentences = open("test-sentences.txt");
out = open("test-sentences.txt.entitylinks", "w");
sents = sentences.readlines();

count = 0;
for i in sents:
	_i = i.strip();
	page = urllib2.urlopen("https://tagme.d4science.org/tagme/tag?lang=en&gcube-token=02801804-90bb-413e-bf67-4763fcad95ec&text=" + urllib.quote_plus(_i)).read()
	out.write(page + "\n");
	count += 1;
	print(count*1.0/len(sents));
out.flush();
out.close();
