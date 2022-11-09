import os
import os.path
from os import path

txt = open("all.txt", "r").read()
filenames = txt.split("\n")

for filename in filenames:
	wavfile = f"{filename}.wav"

	if not path.exists(wavfile):
		print(f"make {wavfile} ...")
		# os.system(f"sox {filename}.mp3 -r 22050 -c 1 {wavfile}")
		os.system(f"lame --decode {filename}.mp3 {wavfile}")
