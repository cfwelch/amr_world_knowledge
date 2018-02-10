
import cPickle as pickle

#sentences = open("test-sentences.txt.tok")
#out = open("test-sentences.txt.general_amr_types", "wb")
#sentences = open("train.txt.sent.tok")
#out = open("train.txt.general_amr_types", "wb")
sentences = open("dev.txt.sent.tok")
out = open("dev.txt.general_amr_types", "wb")
sents = sentences.readlines()

amr_entity_types = {}
amr_entity_types['person'] = ['person', 'family', 'animal', 'language', 'nationality', 'ethnic-group', 'regional-group', 'religious-group', 'political-movement']
amr_entity_types['organization'] = ['organization', 'company', 'government-organization', 'military', 'criminal-organization', 'political-party', 'market-sector', 'school', 'university', 'research-institute', 'team', 'league']
amr_entity_types['location'] = ['location', 'city', 'city-district', 'county', 'state', 'province', 'territory', 'country', 'local-region', 'country-region', 'world-region', 'continent', 'ocean', 'sea', 'lake', 'river', 'gulf', 'bay', 'strait', 'canal', 'peninsula', 'mountain', 'volcano', 'valley', 'canyon', 'island', 'desert', 'forest', 'moon', 'planet', 'star', 'constellation']
amr_entity_types['facility'] = ['facility', 'airport', 'station', 'port', 'tunnel', 'bridge', 'road', 'railway-line', 'canal', 'building', 'theater', 'museum', 'palace', 'hotel', 'worship-place', 'sports-facility', 'market', 'park', 'zoo', 'amusement-park']
amr_entity_types['event'] = ['event', 'incident', 'natural-disaster', 'earthquake', 'war', 'conference', 'game', 'festival']
amr_entity_types['product'] = ['product', 'vehicle', 'ship', 'aircraft', 'aircraft-type', 'spaceship', 'car-make', 'work-of-art', 'picture', 'music', 'show', 'broadcast-program']
amr_entity_types['publication'] = ['publication', 'book', 'newspaper', 'magazine', 'journal', 'natural-object']
amr_entity_types['other'] = ['law', 'treaty', 'award', 'food-dish', 'music-key', 'musical-note', 'variable']
amr_entity_types['thing'] = ['thing']

amr_etype_list = ['nonentity', 'person', 'organization', 'location', 'facility', 'event', 'product', 'publication', 'other', 'thing']

count = 0
output = list()
for i in sents:
	_i = i.decode('utf-8').strip()
	parts = _i.split()
	t_dict = dict()
	line = list()
	for part in parts:
		token = part.lower()
		if token.endswith(";") or token.endswith("."):
			token = token[:-1]
		general_type = "nonentity"
		for nk,nv in amr_entity_types.items():
			if token in nv:
				general_type = nk
				break
		line.append(general_type)
	#print(line)
	output.append(line)
	count += 1
	print(count*1.0/len(sents))

pickle.dump(output, out)
