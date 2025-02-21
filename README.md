# DisCo
A pytorch implementation of paper *Scene Graph Disentanglement and Composition for Generalizable Complex Image Generation*

## Requirements
```
conda env create -n disco python=3.8
conda activate disco
pip install -r requirement.txt
```

## Dataset
Download VisualGenome dataset [here](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) or wherever convenient for you. Then run script `./DisCo/prepare_data/filter_visual_genome.py` and `./DisCo/prepare_data/construct_textual_graph.py` to prepare data.

## Train
```
accelerate launch train_disco.py --use_ema --resolution=512 --batch_size=8 --gradient_accumulation_steps=2 --gradient_checkpointing --max_train_steps=50000 --learning_rate=1e-05  --lr_scheduler="linear" --checkpointing_steps 5000
```