# Adversarial Generative Network (AGN) implementation in PyTorch

Please see the [paper by Sharif et al.](https://arxiv.org/pdf/1801.00349.pdf) for more information.

## Testing your adversarial generator with your webcam

File `video_inference_cv.py` presents a method to test out your adversarial generator (alongside a simple facial recognition classifier) using your webcam.

![testing adversarial generator on webcam](https://github.com/jchaykow/AGN-pytorch/images/webcam_test.png)

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

