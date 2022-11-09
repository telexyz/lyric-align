'''
brew install sox

pip3 install spleeter

sox -t wav -r 22050 -c 1 38313135395f3830_mono.wav -t mp3 38313135395f3830.mp3

sox in.wav -r 22050 -c 1 out.wav

rm done.txt && ls train/*mono* > done.txt && e done.txt
'''

import os
import os.path
from os import path
import re
import json

txt = open("test.txt", "r").read()
filenames = txt.split("\n")

for filename in filenames:

	folder = filename.split("/")[0]
	monofile = f"{filename}_mono.wav"
	vocalfile = f"{filename}/vocals.wav"

	if not path.exists(monofile):
		print(f"make {monofile} ...")
		if not path.exists(vocalfile):
			cmd = f"spleeter separate -o {folder}/ {filename}.wav"
			print(cmd)
			os.system(cmd)
		os.system(f"sox {vocalfile} -r 22050 -c 1 -b 32 {monofile}")
	
	mp3file = f"main/data/{filename}.mp3"
	if not path.exists(mp3file):
		cmd = f"sox -t wav -r 22050 -c 1 {monofile} -t mp3 {mp3file}"
		print(cmd)
		os.system(cmd)

	
# txt = open("train.txt", "r").read()
# filenames = txt.split("\n")

# for filename in filenames:
# 	# filename = "train/37393039325f3233"
# 	data = json.load(open(f"{filename}.json"))
# 	with open(f"main/data/{filename}.lab", 'w') as f:
# 		for line in data:
# 			for word in line['l']:
# 				s = word['s'] / 1000
# 				e = word['e'] / 1000
# 				w = word['d']
# 				# 0.000000	0.141169	moving
# 				# 1.300244	1.961511	forward
# 				f.write(f"{s}\t{e}\t{w}\n")

# with open("lines_train.txt", 'w') as f:
# 	for filename in filenames:
# 		txt = open(f"main/data/{filename}.txt", "r").read().lower()
# 		f.write(txt)
# 		f.write("\n")
