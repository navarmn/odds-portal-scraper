"""
Run the Odds Portal scraping suite, processing all the present soccer league
JSON files in lexicographical order.
"""

import os
from os import listdir, sep
from os.path import isfile, join
from Scraper import Scraper

import logging

filename_log = '.logfile.log'
if os.path.exists(os.path.join(os.path.dirname(__file__), filename_log)):
    os.remove(os.path.join(os.path.dirname(__file__), filename_log))
    
logging.basicConfig(filename='.logfile.log', level=logging.DEBUG)

soccer_match_path = "." + sep + "leagues" + sep + "soccer"

initialize_db = True

for possible_file in listdir(soccer_match_path):
    logging.debug("================================== \n")
    logging.debug("File is: {} \n".format(possible_file))

    if isfile(join(soccer_match_path, possible_file)):
        soccer_match_json_file = join(soccer_match_path, possible_file)
        logging.debug("soccer JSON file: {} \n".format(soccer_match_json_file))

        with open(soccer_match_json_file, "r") as open_json_file:
            json_str = open_json_file.read().replace("\n", "")
            logging.debug("Starting scrapping....")
            match_scraper = Scraper(json_str, initialize_db)
            match_scraper.scrape_all_urls(True)
            if initialize_db is True:
                initialize_db = False