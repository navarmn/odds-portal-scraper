from os import listdir, sep
from os.path import isfile, join

soccer_match_path = '.' + sep + 'leagues' + sep + 'soccer'

for possible_file in listdir(soccer_match_path):
    if isfile(join(soccer_match_path, possible_file)):
        soccer_match_json_file = join(soccer_match_path, possible_file)
        with open(soccer_match_json_file, 'r') as open_json_file:
            json_str = open_json_file.read().replace('\n', '')
            # TODO make Scrapers