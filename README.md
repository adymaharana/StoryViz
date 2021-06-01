## Improving Generation and Evaluation of Visual Stories via Semantic Consistency

PyTorch code for the NAACL 2021 paper "Improving Generation and Evaluation of Visual Stories via Semantic Consistency".
Link to arXiv paper: [https://arxiv.org/abs/2105.10026](https://arxiv.org/abs/2105.10026)

#### Requirements:
This code has been tested on torch==1.7.1 and torchvision==0.8.2

#### Prepare Repository:
 Download the PororoSV dataset and associated files from [here](https://drive.google.com/file/d/1BqKizOZn4o4dbwNGK7ThCDnNEOVAolnf/view?usp=sharing) and save it as ```./data```.
 Download GloVe embeddings (glove.840B.300D) from [here](https://nlp.stanford.edu/projects/glove/). The default location of the embeddings is ```./data/``` (see ```./dcsgan/miscc/config.py```). 

#### Training DuCo-StoryGAN:

To train DuCo-StoryGAN, first train the VideoCaptioning model on the PororoSV dataset:\
```python train_mart.py --data_dir <path-to-data-directory>```\
Default parameters were used to train the model used in our paper.

Next, train the generative model:\
```python train_gan.py --cfg ./cfg/pororo_s1_duco.yml --data_dir <path-to-data-directory>```\
If training DuCo-StoryGAN on a new dataset, make sure to train the Video Captioning model (see below) before training the GAN. The vocabulary file prepared for the video-captioning model is re-used for generating common ```input_ids``` for both models. Change location of video captioning checkpoint in config file.
    
Unless specified, the default output root directory for all model checkpoints is ```./out/```


#### Training Evaluation Models:
* Video Captioning Model\
The video captioning model trained for DuCo-StoryGAN (see above) is used for evaluation.
```python train_mart.py --data_dir <path-to-data-directory>```

* Hierarchical Deep Multimodal Similarity (H-DAMSM)\
```python train_damsm.py --cfg ./cfg/pororo_damsm.yml --data_dir <path-to-data-directory>```
    
* Character Classifier\
```python train_classifier.py --data_dir <path-to-data-directory> --model_name inception --save_path ./models/inception --batch_size 8 --learning_rate 1e-05```


### Acknowledgements
The code in this repository has been adapted from the [MART](https://github.com/jayleicn/recurrent-transformer), [StoryGAN](https://github.com/yitong91/StoryGAN) and [MirrorGAN](https://github.com/qiaott/MirrorGAN) codebases.