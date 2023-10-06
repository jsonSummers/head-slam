#!/bin/bash
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip
unzip MH_01_easy.zip -d data
mv data/mav0/cam0/sensor.yaml data/mav0/cam0/data
mv data/mav0/cam0/data data/mav0/cam0/euroc
mv data/mav0/cam0/euroc data/euroc
rm -r data/mav0
rm MH_01_easy.zip
