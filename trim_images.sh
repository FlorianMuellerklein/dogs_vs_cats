#!/bin/bash

IMAGES="/Volumes/Mildred/Kaggle/DogsvsCats/data/train/*.jpg"
for file in $IMAGES
do
	echo "$file"
	convert $file -resize "192x192^" -gravity center -crop 192X192+0+0 +repage $file

done
