# Image 2 Poem

This repository contains code for image to poem generator. For setting up the code follow the instructions.

## Download repository and environment setup

```
git clone https://github.com/ThakurAbhi7/Image2Poem.git
cd Image2Poem
conda env create -f <path_to_yaml_file>
conda activate I2P
```

## For training model

```
python trainer.py --checkpoint_poem checkpoints/poem --checkpoint_title checkpoints/title
```

## For testing model

```
python generate_poem.py --path "path to image" --checkpoint_poem checkpoints/poem/pytorch_model.bin --checkpoint_title checkpoints/title/pytorch_model.bin
```