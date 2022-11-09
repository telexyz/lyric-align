'''
[{
	"s": 0, "e": 0, 
	"l": [
		{"s": 0, "e": 0, "d": "Tôi"}, 
		{"s": 0, "e": 0, "d": "xin"}, 
		{"s": 0, "e": 0, "d": "gửi"}, 
		{"s": 0, "e": 0, "d": "về"}
	]
}]

pip3 install mutagen
'''

import os
import os.path
from os import path
import re
import json

from mutagen.mp3 import MP3
# audio = MP3("example.mp3")
# print(audio.info.length)


txt = open("submit.txt", "r").read()
filenames = txt.split("\n")

for filename in filenames:
	print(f"cheating {filename} ...")
	audio = MP3(f"{filename}.mp3")
	length = audio.info.length
	length_ms = int(length * 1000)
	txt = open(f"{filename}.txt").read()
	# print(txt)
	words =re.split("\\s+", txt)
	n = len(words)
	print(length, length_ms, n)
	# print(words)
	data =[{ "s": 0, "e": length_ms, "l": []}]
	# data["l"] = [{"s": 0, "e": 0, "d": "về"}, ..]
	l = []
	step = length_ms // n
	start = 0
	end = step
	for w in words:
		label = {"s": start, "e": end, "d": w}
		start = end
		end += step
		l.append(label)
	data[0]["l"] = l
	print(data)
	with open(f"{filename}.json", 'w') as f:
		f.write(json.dumps(data))
