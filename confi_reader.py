from configparser import ConfigParser
import os

parser = ConfigParser()
parser.read('config.ini')

# print(parser.sections())
a = (parser.get('tipos','tipos')).split(",")
print(a[0])

# print(parser.options('modelos'))