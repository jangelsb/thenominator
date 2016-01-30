
import importlib

def reload(nom):
	importlib.reload(nom)


def grabAllFromYears(fromYear, toYear):

	import tmdbsimple as tmdb
	tmdb.API_KEY = 'YOUR_API_KEY_HERE'

	search = tmdb.Discover()
	for year in range(fromYear, toYear+1):

		r = search.movie(primary_release_year = year)
		numOfPages = r['total_pages']

		print("Accessing Year: {}".format(year))
		print("Number of pages: {}".format(numOfPages))

		#new file per year
		text_file = open("{}.txt".format(year), "w")

		for i in range(1, r['total_pages']+1):
			print("Page: {}/{}".format(i,numOfPages))
			r = search.movie(primary_release_year = year, page = i)
			for a in r['results']:
				print("{}".format(a['title']), file=text_file)

		text_file.close()

def grabAllFromYear(year):
	grabAllFromYears(year, year)


	


