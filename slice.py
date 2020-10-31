import re

f = open("news_test.txt", 'r')
string = f.read()
string = string.split('\n\n')
print(string[1])