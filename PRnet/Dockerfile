FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt update && \ 
    apt install -y gcc python3-dev python3-pip libxml2-dev libxslt1-dev zlib1g-dev g++ git fish cmake build-essential vim htop openssh-server tree tmux  
RUN pip install pip -U && \ 
    pip install numpy pandas scikit-learn scikit-image matplotlib tqdm scanpy rdkit anndata seaborn 

