# head-slam

Dataset download [here](http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip)

data structure:

head-slam  
-src  
-data/  
--euroc/  
---insert cam0 images here and sensor.yaml

Example:
python main.py euroc

Install:
git clone https://github.com/jsonSummers/head-slam.git
cd head-slam
chmod +x download_dataset.sh
conda env create -f environment.yaml
conda activate head-slam
