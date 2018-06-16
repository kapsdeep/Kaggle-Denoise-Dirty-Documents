# Kaggle-Denoise-Dirty-Documents

# Problem statement
#### Develop ML solution on Optical character recognition - a technique to retrieve old printed text (with stains, tea marks, sun spots) into digital format for better accessibility. 
* _Complexity: medium (since 3 years old)_
* _Dataset: scanned pages with synthetic noise added_
* _Objective: rmse=0.0275 b/w pixel intensities of cleaned & actual image_
* _Deliverable: Keras implementation using winograd_
* _Timeline: 06/16 AM IST_

## Submission file
[DEEPAK_KAPOOR_Session7.ipynb](https://github.com/kapsdeep/EIP-4B/blob/master/Deepak_BATCH_6_ASSIGNMENT4B.ipynb)

## Strategy Plan

| Plan        | AIs          | Status  | Comments  |
  | ------------- |:-------------:| -----:| -----:|
  | Revise notes on winograd      | Try computation load reduction using winograd | Not started | Try in parallel tracks |
  | Read available papers      | [Kaggle: Denoising Dirty Documents with MATLAB - File Exchange - MATLAB Central](https://in.mathworks.com/matlabcentral/fileexchange/51812-kaggle--denoising-dirty-documents-with-matlab) [Image Processing + Machine Learning in R: Denoising Dirty Documents Tutorial Series No Free Hunch](http://blog.kaggle.com/2015/12/04/image-processing-machine-learning-in-r-denoising-dirty-documents-tutorial-series/)  [Colin Priest](https://colinpriest.com/2015/08/01/denoising-dirty-documents-part-1/)  & [Denoising with Random Forests - 0.02628](https://www.kaggle.com/johnnystrings/denoising-with-random-forests) [Denoising stained images using kmeans clustering – Corpocrat Magazine](https://corpocrat.com/2015/07/20/noise-removal-using-kmeans-on-stains-in-scanned-images/)     [Location-aware kernels for large images tiled processing with neural networks](https://medium.com/shallowlearnings/spatial-tiled-processing-for-large-images-with-convolutional-neural-networks-3936ed7aebec) [(PDF) Enhancement and Cleaning of Handwritten Data by Using Neural Networks](https://www.researchgate.net/publication/221258673_Enhancement_and_Cleaning_of_Handwritten_Data_by_Using_Neural_Networks) [kaggle/denoising-dirty-documents at master · gdb/kaggle · GitHub](https://github.com/gdb/kaggle/tree/master/denoising-dirty-documents)| Initiate |  Pull out best practices |
| Check top kaggle submissions | PFB | Initiated |  Pull out best practices |
| Devise project plan | PFB | Completed |   Update on need basis |

#### Top kaggle submissions to read
| Kaggler        | Score          | Kernel  | Packages  |
  | ------------- |:-------------:| -----:| -----:|
  | Alexey Antonov      | 0.02602 | [aantonov Kaggle](https://www.kaggle.com/antonov/kernels) | "as in NN_Starter Kit by Rangel Dokov" numpy as np, theano, theano.tensor as T, cv2, os, itertools, , theano.config.floatX = 'float32'     |
  | Johny Strings      | 0.02602 | [Johnny Strings Kaggle](https://www.kaggle.com/johnnystrings/kernels) | same as Oliver Sherouse     |
  | Colin      | 0.00512 | [Colin Kaggle](https://www.kaggle.com/colinpriest/kernels?sortBy=voteCount&group=everyone&pageSize=20&userId=46478&language=R) | Implemented in R    |
  | Rangel Dokov      | 0.02682 | [Background removal Kaggle](https://www.kaggle.com/rdokov/background-removal/code) | """  Simple background removal code    __author__ : Rangel Dokov    The basic idea is that we have a foreground object of interest (the dark text)  and we want to remove everything that is not part of this foreground object.    This should produce results somewhere around 0.06 on the leaderboard.  """  import numpy as np  from scipy import signal  from PIL import Image |
  | Oliver Sherouse      | 0.02811 | [Let's Try Simple Linear Regression Kaggle](https://www.kaggle.com/oliversherouse/let-s-try-simple-linear-regression/code) | concurrent.futures,csv, logging,random, joblib,numpy as np, sklearn.ensemble, sklearn.cross_validation, sklearn.metrics, skimage.data, from pathlib import Path, from PIL import Image as image     |

## Project Plan
>Lets break the project into independent tasks to enable deeper understanding of individual topics & explore multiple ways to reach the objective

| Project        | ETA          | Risk(s)  | rmse  | Analysis  | Status  |
  | ------------- |:-------------:| -----:| -----:| -----:| -----:|
  | Prj#1: Kaggle competitions using google collab       | 6/16 | None | 0.2 | Worst error without any processing | Done  :white_check_mark: |
| Prj#2: Re-use Prj#1 with "simple background removal model"       | 6/16 | None | 0.06002 | Median filters are used for salt & pepper noise & results point out that a image processing would be helpful for the challenge | Done  :white_check_mark: |
| Prj#3: Build over Prj#2 with "simple linear regression model"       | 6/18 | Monday is working day | NA | NA | Initiated     |
| Prj#4: Build over Prj#2 using image processing techniques from top kagglers       | 6/20 | NA | NA | NA | Not started     |
| Prj#5: TBD       | 6/22 | NA | NA | NA | Not started     |
| Prj#6: TBD       | 6/25 | NA | NA | NA | Not started     |

#### Prj#1: Kaggle competition using google collab 
1. Enable train/test access in collaboratory :ballot_box_with_check:
    - #1: Mount google drive to collab
      - [Importing data to Google Colaboratory](https://steemit.com/google-colab/@ankurankan/importing-data-to-google-colaboratory)
      - [Mount gdrive to collab using FUSE](https://colab.research.google.com/drive/1srw_HFWQ2SMgmWIawucXfusGzrj1_U0q)
    - #2: Upload data to google drive & download to collab
        - googledrive APIs
        - Kaggle APIs
          - [Official Kaggle API](https://github.com/Kaggle/kaggle-api)
        - Pydrive (since Byte Order Marker can be removed):heavy_check_mark:
          -  [GitHub & BitBucket HTML Preview](https://htmlpreview.github.io/?https://raw.githubusercontent.com/gsuitedevs/PyDrive/master/docs/build/html/quickstart.html)
2. Play with image using matplotlib & PIL :ballot_box_with_check:
    - [Pyplot Image tutorial](https://matplotlib.org/users/image_tutorial.html) 
    - [Tutorial — Pillow (PIL Fork) 5.1.1 documentation](https://pillow.readthedocs.io/en/5.1.x/handbook/tutorial.html)
3. Develop error loss function (TBD)
4. Develop kaggle submission function (download from collab):heavy_check_mark:

#### Prj#2: Re-use Prj#1 with "simple background removal model"
1. Try background removal model to hit rsme: 0.06:heavy_check_mark: 
2. Read through scipy:signal & PIL:Image:heavy_check_mark:
3. Read about Image filtering
    - [CS6670: Computer Vision Lecture 2: Image filtering](https://www.cs.cornell.edu/courses/cs6670/2011sp/lectures/lec02_filter.pdf)
    - [CSE.USF Image Filtering](http://www.cse.usf.edu/~r1k/MachineVisionBook/MachineVision.files/MachineVision_Chapter4.pdf)
    - [Auckland Univ: Image Filtering Median Filtering](https://www.cs.auckland.ac.nz/courses/compsci373s1c/PatricesLectures/Image%20Filtering_2up.pdf)
    - [Signal Processing (scipy.signal) — Other filters Median Filter](https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html)
    - [Prepress and File Formats: Raster vs Vector](https://www.slideshare.net/JenniferJanviere/prepress)

#### Prj#3: Build over Prj#2 using simple linear regression model
1. Try adaptive thresholding
2. Try canny edge detection

#### Prj#4: Build over Prj#2 using image processing techniques from top kagglers
1. Try adaptive thresholding
2. Try canny edge detection


## Solution summary
* In-progress

_Note: TBD_

References
- [Complete list of github markdown emoji markup · GitHub](https://gist.github.com/rxaviers/7360908)
- #https://colab.research.google.com/notebooks/io.ipynb#scrollTo=zU5b6dlRwUQk
- #Alternate: https://github.com/Kaggle/kaggle-api (didn't try kaggle APIs though seemingly easy)
- #Alternate: https://medium.com/@likho2manish/dev-steps-to-google-colab-5c72779c0ae9 (didn't try googledrive APIs)
- #Pydrive: https://github.com/gsuitedevs/PyDrive/tree/master/docs 
- #Pydrive: https://htmlpreview.github.io/?- -https://raw.githubusercontent.com/gsuitedevs/PyDrive/master/docs/build/html/filemanagement.html
- #Options: https://steemit.com/google-colab/@ankurankan/importing-data-to-google-colaboratory
