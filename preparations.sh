## Create a virtual environment and get all dependencies
python -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --upgrade setuptools
python -m pip install -r DisCo/requirements.txt # Successfully installed accelerate-1.3.0 cachetools-5.5.1 charset-normalizer-3.4.1 diffusers-0.32.2 einops-0.8.0 google-auth-2.38.0 google-auth-oauthlib-1.2.1 h5py-3.12.1 huggingface-hub-0.28.1 importlib_metadata-8.6.1 importlib_resources-6.5.2 nvidia-cusparselt-cu12-0.6.2 opencv-python-4.11.0.86 regex-2024.11.6 requests-oauthlib-2.0.0 rsa-4.9 safetensors-0.5.2 tokenizers-0.21.0 torch-2.6.0 torchaudio-2.6.0 torchvision-0.21.0 tqdm-4.67.1 transformers-4.48.2 triton-3.2.0 zipp-3.21.0
python -m pip install nvidia-nccl-cu11
cp ./.venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so.12 ./.venv/lib/python3.12/site-packages/nvidia/nccl/lib/

## Get the Visual Genome dataset
mkdir -p VisualGenome/clip/VG_100K
mkdir -p VisualGenome/clip/VG_100K_2
cd VisualGenome
wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip
wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip
wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip
wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip
# wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/scene_graphs.json.zip
wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/object_alias.txt
wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationship_alias.txt
unzip \*.zip
rm *.zip
cd ..

## Filter and prepare the dataset
python ./DisCo/prepare_data/filter_visual_genome.py --splits_json './DisCo/prepare_data/vg_splits.json'
python -m DisCo.prepare_data.construct_textual_graph

## Get the Stable Diffusion model
python download_stable_diffusion.py