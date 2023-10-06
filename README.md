# head-slam

Install:
git clone https://github.com/jsonSummers/head-slam.git  
cd head-slam  
chmod +x download_dataset.sh  

### Creating the environment
conda create -n head-slam python=3.8 numpy=1.23 -y  
conda activate head-slam  
conda install -c "conda-forge/label/cf201901" opencv -y  
python3 -m pip install pygame  
python3 -m pip install pyyaml  


conda env create -f environment.yaml  
conda activate head-slam  

Example:
python main.py euroc
