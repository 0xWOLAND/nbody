name=$1; 
title=$2;
ffmpeg -r 2 -pattern_type glob -i "$name*.png" -vf "scale='min(1280,iw)':-2,format=yuv420p" -codec:v mpeg4 -q:v 5 -r 30 -pix_fmt yuv420p evolution_$title.mp4 
