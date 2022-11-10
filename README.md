https://github.com/jhuang448/LyricsAlignment-MTL

# LyricsAlignment-MTL

This repository consists of code of the following paper:

Jiawen Huang, Emmanouil Benetos, Sebastian Ewert, "**Improving Lyrics Alignment through Joint Pitch Detection**," 
International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2022. [https://arxiv.org/abs/2202.01646](https://arxiv.org/abs/2202.01646)

## Dependencies

This repo is written in python 3. Pytorch is used as the deep learning framework. To install the required python packages, run

```
pip3 install -r requirements.txt
```

Besides, you might want to install some source-separation tool (e.g. [Spleeter](https://github.com/deezer/Spleeter), [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch)) or use your own system to prepare source-separated vocals.

## Usage

Check the [notebook](https://github.com/jhuang448/LyricsAlignment-MTL/blob/main/example.ipynb) for a quick example.


## Training

### The baseline acoustic model (**Baseline**)

```
python train.py --checkpoint_dir=/where/to/save/checkpoints/ 
                --log_dir=/where/to/save/tensorboard/logs/ 
                --model=baseline --cuda
```

Run `python train.py -h` for more options.

## Inference

The following script runs alignment using a pretrained baseline model without boundary information (**Baseline**):

```
python eval.py --load_model=./checkpoints/checkpoint_Baseline 
               --pred_dir=/where/to/save/predictions/
               --model=baseline --cuda
```

The following script runs alignment using the pretrained MTL model with boundary information (**MTL+BDR**) on Jamendo:

```
python eval_bdr.py --load_model=./checkpoints/checkpoint_MTL 
                   --pred_dir=/where/to/save/predictions/
                   --bdr_model=./checkpoints/checkpoint_BDR
                   --model=MTL --cuda
```

The generated csv files under `pred_dir` can be easily evaluated using the evaluation script in [jamendolyrics](https://github.com/f90/jamendolyrics).
