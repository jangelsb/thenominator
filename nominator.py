# This uses the omdb and tmdb apis to generate a text file with
# a list of movies from a given year. It can also generate text files for a range of years.

# Pro tip:
# to load file into interpreter 
# exec(open('nominator.py').read())

import importlib

import omdb as omdb
from keys import tmdb_apikey
import tmdbsimple as tmdb
tmdb.API_KEY = tmdb_apikey #set this to your TMDB api key

def reload(module):
	"""
	Reloads the module (the nominator.py) into the interpreter 

	Returns
	-------
	The module object

	Example
	-------
	>>> import nominator as nom
	>>> nom.reload(nom)
	"""

	importlib.reload(module)


def grabAllFromYears(fromYear, toYear):
	"""
	Generates a text file with list of movies from fromYear to toYear.

	Returns
	-------
	Nothing

	Example
	-------
	>>> grabAllFromYears(2009,2015)
	"""
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

def grabNFromYears(fromYear, toYear, n):
	"""
	Generates a text file with list of movies from fromYear to toYear 
	but only grabs the first n of each year or all of them if there is 
	under n movies for that year.

	Returns
	-------
	Nothing

	Example
	-------
	>>> grabAllFromYears(2009,2015)
	"""
	search = tmdb.Discover()
     
 
	for year in range(fromYear, toYear+1):

		r = search.movie(primary_release_year = year)
		numOfPages = r['total_pages']

		print("Accessing Year: {}".format(year))
		print("Number of pages: {}".format(numOfPages))

		#new file per year
		text_file = open("{}.txt".format(year), "w")

		count = 0
		for i in range(1, r['total_pages']+1):

			if count > n:
				break

			print("Page: {}/{}".format(i,numOfPages))
			r = search.movie(primary_release_year = year, page = i)
			for a in r['results']:
				print("{}".format(a['title']), file=text_file)
				count+=1

		text_file.close()

def grabAllFromYear(year):
	"""
	Generates a text file with list of movies from the given year.

	Returns
	-------
	Nothing

	Example
	-------
	>>> grabAllFromYears(1994)
	"""

	grabAllFromYears(year, year)


def getMovieData(title, year = None):
	res = omdb.request(t=title, y=year)
	return res.content


