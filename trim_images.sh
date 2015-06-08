#!/bin/bash

IMAGES="/Volumes/Mildred/Kaggle/DogsvsCats/data/test1/*.jpg"
for file in $IMAGES
do
	echo "$file"
	convert $file -resize 192x192! -gravity center $file

done
