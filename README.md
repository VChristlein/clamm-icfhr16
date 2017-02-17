# clamm-icfhr16
Code written for the ICFHR16 competition on the classification of medieval handwritten script

Estimates for each image of the input folder the containing script type and
writes all estimations in the files 'belonging_class.csv' and
'belonging_matrix.csv' (written to the current folder)


## Prerequesits:

Python Packages needed: os, glob, cv2, numpy, multiprocessing, gzip, cPickle, scipy, sklearn, zca

Necessary files in the `trained` folder:
- pretrained principal component analysis (pca.pkl.gz)
- pretrained Gaussian mixture model (gmm.pkl.gz)
- pretrained total-variability space (tv.pkl.gz)
- pretrained within-class-covariance-normalization (wccn.pkl.gz)
- pretrained support vector machines (svm.pkl.gz)
- optional: pretrained linear discriminant analysis (lda.pkl.gz)
  - Note: in the competition lda was used for task2, however note that svm achieves
    better results

## Usage
`python2 <inputdir>`

where <inputdir> is the folder of the input images. 

You can also evaluate on your own dataset using a labelfile, see '-h' for more
possible options.

