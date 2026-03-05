# AgeMoE
This is an anti-aging small molecule prediction model based on MoE.<br>
# Install Requirements
You can manually add the environment in the following ways. The version about ```pytorch``` and ```cudatoolkit``` should be depended on your machine.<br>
```
conda create -n AgeMoE python=3.10 \
conda activate AgeMoE \
pip3 install torch==2.4.0+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 \
pip install numpy==1.24.3 \
pip install pyyaml==6.0.3 \
pip install scanpy \
pip install pandas==2.3.3 \
pip install torch-geometric \
conda install rdkit=2025.9.1 -c rdkit \
```
We also provide the corresponding environment file, You can reproduce the environment directly through the provided .yaml file.<br>
```
conda env create -f environment.yaml
```
To ensure the accurate input of the model's transcription information, the Lincs_L1000.h5ad file needs to be placed in the ```lincs_cache``` folder. This file can be downloaded from https://drive.google.com/file/d/1Z05yalquBrXG4sVitkspbvScFFXee7DJ/view?usp=sharing.
