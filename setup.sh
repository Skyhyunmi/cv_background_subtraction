#!/bin/sh
mkdir /tmp/cv_stab
mkdir result_es
echo "Downloading Dataset ..."
curl http://jacarini.dinf.usherbrooke.ca/static/dataset/cameraJitter/badminton.zip > /tmp/cv_stab/zip.zip
unzip -qq /tmp/cv_stab/zip.zip
echo "Moving Groundtruth to main folder ..."
mv badminton/groundtruth ./groundtruth
echo "Moving Input to main folder ..."
mv badminton/input ./input
rm -rf badminton
echo "Job done."