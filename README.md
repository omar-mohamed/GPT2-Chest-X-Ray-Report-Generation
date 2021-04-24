# GPT2-Chest-X-Ray-Report-Generation (CDGPT2)
This is the implementation of the CDGPT2 model mentioned in our paper 'Automated Radiology Report Generation using Conditioned Transformers'.

Paper link [here](https://doi.org/10.1016/j.imu.2021.100557).

We automatically generate full radiology reports given chest X-ray images from the IU-X-Ray dataset by conditioning a pre-trained GPT2 model on the visual and semantic features of the image.

Model checkpoint [here](https://drive.google.com/drive/folders/1GRT5-aQ7WXN9F7OzDjl3aHLGERT63sIh?usp=sharing).
![chest overall dpi](https://user-images.githubusercontent.com/6074821/113484358-320c1000-94a8-11eb-83da-fc2ca2ca4e86.png)

## Sample Predictions
![image](https://user-images.githubusercontent.com/6074821/113487044-098b1280-94b6-11eb-93b0-f2bf3202010f.png)

## Installation & Usage

- pip install -r requirements.txt
- nlg-eval --setup
- python train.py

## Related Repositories
- VSGRU repository [here](https://github.com/omar-mohamed/X-Ray-Report-Generation).
- Finetuned Chexnet repository [here](https://github.com/omar-mohamed/Chest-X-Ray-Tags-Classification).

## Citation
To cite this paper, please use:

```
@article{ALFARGHALY2021100557,
title = {Automated radiology report generation using conditioned transformers},
journal = {Informatics in Medicine Unlocked},
volume = {24},
pages = {100557},
year = {2021},
issn = {2352-9148},
doi = {https://doi.org/10.1016/j.imu.2021.100557},
url = {https://www.sciencedirect.com/science/article/pii/S2352914821000472},
author = {Omar Alfarghaly and Rana Khaled and Abeer Elkorany and Maha Helal and Aly Fahmy}
}
```
