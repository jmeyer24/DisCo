from huggingface_hub import snapshot_download

# snapshot_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5", ignore_patterns=["*.ckpt", "*.safetensors"], local_dir="./StableDiffusion")
snapshot_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5", local_dir="../StableDiffusion")