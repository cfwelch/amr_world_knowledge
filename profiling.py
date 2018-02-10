import sys,pstats

def main():
	f_name = sys.argv[1];
	p = pstats.Stats(f_name);
	p.strip_dirs().sort_stats(1).print_stats(20);

if __name__ == "__main__":
	main();
