# Adversarial Generative Network (AGN) implementation in PyTorch

Please see the [paper by Sharif et al.](https://arxiv.org/pdf/1801.00349.pdf) for more information.

## Testing your adversarial generator with your webcam

File `video_inference_cv.py` presents a method to test out your adversarial generator (alongside a simple facial recognition classifier) using your webcam.

![testing adversarial generator on webcam](https://github.com/jchaykow/AGN-pytorch/blob/master/images/webcam_test.png)


## Overview of Methods

### Goal

to generate eyeglass frames that make a facial recognition classifier think you are someone else

### Tools

- 1 GAN
- 1 classifier
- 1 big dataset of eyeglass frames 
- some images of celebrity faces 
- some images of my face

### Algorithm

Please see paper for more info, but here is my brief overview of the Algorithm 1 from the paper:

![intrepreting the algorithm](https://github.com/jchaykow/AGN-pytorch/blob/master/images/agn_algorithm_translate.png)

### Finetuning the Facial Recognition Network

![overview](https://github.com/jchaykow/AGN-pytorch/blob/master/images/finetune_face_classifier_overview.png)

![overview](https://github.com/jchaykow/AGN-pytorch/blob/master/images/finetuning_loss.png)

### Pre-train GAN

![overview](https://github.com/jchaykow/AGN-pytorch/blob/master/images/gan_overview.png)

![static](https://github.com/jchaykow/AGN-pytorch/blob/master/images/static.png)

![shape](https://github.com/jchaykow/AGN-pytorch/blob/master/images/shape.png)

![background](https://github.com/jchaykow/AGN-pytorch/blob/master/images/background.png)

![color variation](https://github.com/jchaykow/AGN-pytorch/blob/master/images/color_variation.png)

![lighting variation](https://github.com/jchaykow/AGN-pytorch/blob/master/images/lighting_variation.png)

## Prepare Data for AGN Training

![resize and align from iPhone video](https://github.com/jchaykow/AGN-pytorch/blob/master/images/resize_and_align.png)

![transforms applied to images](https://github.com/jchaykow/AGN-pytorch/blob/master/images/transforms.png)

![facial landmark analysis for placing glasses](https://github.com/jchaykow/AGN-pytorch/blob/master/images/facial_landmark_analysis.png)

## AGN Training

![two updates to generator](https://github.com/jchaykow/AGN-pytorch/blob/master/images/two_updates.png)


## For video pre-processing steps

1. 

`brew install imagemagick`

```
for szFile in ./Michael_Chaykowsky/*.png
do 
    magick mogrify -rotate 90 ./Michael_Chaykowsky/"$(basename "$szFile")" ; 
done
```

1. 

`pip install autocrop`
   
`$ autocrop -i ./me_orig/Michael_Chaykowsky -o ./me/Michael_Chaykowsky160 -w 720 -H 720 --facePercent 80`

Directory must have subdirectory with name of person and images stored inside
- this is already done by the previous autocrop step

1. 

`! pip install tensorflow==1.13.0rc1`

`! pip install scipy==1.1.0`

Use the directory above the classes directory (not the actual directory holding the images)

```
for N in {1..4}; do \
python ~/Adversarial/data/align/align_dataset_mtcnn.py \ # tensorflow script
~/Adversarial/data/me/ \ # current directory
~/Adversarial/data/me160/ \ # new directory
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 0.25 \
& done
```

```
with open('bboxes_fnames_test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(['filename']))
    a = [os.path.basename(x) for x in glob.glob('agn_me_extras160/Michael_Chaykowsky/*.png')]
    writer.writerows(zip(a))
```

## Directory structure

```
project
│   README.md
│   AGN.ipynb  
│
└───data
│   │   files_sample.csv
│   └───eyeglasses
│   │
│   └───test_me
│       └───train
|           └───Adrien_Brody
|           ...
|           └───Michael_Chaykowsky
|           ...
|           └───Venus_Williams
│       └───val
|           └───Adrien_Brody
|           ...
|           └───Michael_Chaykowsky
|           ...
|           └───Venus_Williams
│   
└───models
│   │   inception_resnet_v1.py
│   │   mtcnn.py
│   └───utils
```

## Refrences

@article{Sharif_2019,
   title={A General Framework for Adversarial Examples with Objectives},
   volume={22},
   ISSN={2471-2574},
   url={http://dx.doi.org/10.1145/3317611},
   DOI={10.1145/3317611},
   number={3},
   journal={ACM Transactions on Privacy and Security},
   publisher={Association for Computing Machinery (ACM)},
   author={Sharif, Mahmood and Bhagavatula, Sruti and Bauer, Lujo and Reiter, Michael K.},
   year={2019},
   month={Jul},
   pages={1–30}
}

@misc{cao2018vggface2,
      title={VGGFace2: A dataset for recognising faces across pose and age}, 
      author={Qiong Cao and Li Shen and Weidi Xie and Omkar M. Parkhi and Andrew Zisserman},
      year={2018},
      eprint={1710.08092},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

