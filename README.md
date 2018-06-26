# Problem statement
  #### Develop ML solution to remove noise from scanned copies of old book pages/handwritten documents (marked with stains, tea marks, sun spots - in the challenge the noise is synthetic though). Denoised scanned pages are helpful for Optical character recognition - a technique to convert old printed text into digital format for better accessibility. 
  * Complexity: medium (since 3 years old)
  * Dataset: scanned pages with synthetic noise added - (train, train_cleaned), test
  * Objective: rmse=0.0275 b/w pixel intensities of cleaned & actual(ground_truth) image
  * Deliverable: Keras implementation using winograd
  * Submission file: [DEEPAK_KAPOOR_Session7.ipynb](https://github.com/kapsdeep/Kaggle-Denoise-Dirty-Documents/blob/master/v4_0_DEEPAK_KAPOOR_Session7.ipynb)

  ## More about NOISE
  >Noise is random variation of image density visible as pixel level variations in digital images ( and as grains in films). Noise is equally important as "Sharpness" to define image quality.

  Signal to Noise ratio (SNR) is widely used metric to define image quality since the noise is closely related to signal value.
  Noise measurments generally refer to Root mean square error(RMSE) 
  ```
  N = σ(S), where σ denotes the standard deviation. (S can be the signal in any one of several channels: R, G, B, Y)
  ```

  Types of Noise(s)
  - Temporal Noise is one which varies each time a image is captured/generated
      - can be reduced by signal averaging i.e. sum(N)/N
  - Spatial Noise is fixed pattern noise appearing over the image space

  Keras functions for temporal/spatial noise
  - MaxPooling1D: Max pooling operation for temporal data
  - MaxPooling2D: Max pooling operation for spatial data
  - MaxPooling3D: Max pooling operation for 3D data (spatial or spatio-temporal)

## Solution summary
  Risks:small_red_triangle:
  1. ~~selecting b/w scikit solution & autoencoder~~ will document scikit if nothing else otherwise autoencoder still looks hopefull
  2. developing solution to map pixel values to images from ndarray
  3. ~~improve text sharpness~~ deeper layers, less upsampling2D/MaxPooling helped regain text clarity by reducing interpolation
  4. implement ensemble
  
  Progress :clock130:
  * The project evaluation criteria states that model performance will be graded by rmse on pixel intensities values b/w (predicted image, ground_truth image). 
  * Drawing inpiration from sample submission file, I plan to use flattened arrays (pixel id, pixel intensity value) of (train-images, train_cleaned-images) & try ML(DL)/Image processing techniques to optimize.
  * Worst case score is 0.2 derived by submitting csv file using test data without any processing hence my first goal is to surpass that
      * I could hit 0.06(Prj#2, pfb) with simple median filter & ML/DL are still waiting :grinning:
  * Dropping idea of linear regression since rmse score isn't improving beyond 0.7
  * Autoencoder model is working better than simple median filter & thresholding
    * though the results are visibly better but when I kaggle evaluation score dropped to 0.11479 since the csv contained results from autoencoder for 540x420 images & Prj#2 results for 258x540 images. Additionally, I am not sure whether pixel intensities were mapped correctly in the submission.csv
  * To cover the risk of running out of time, I used another solution based on random forrest using scikit, it seems to be working rightaway & giving rmse scrore=0.02656
    * Analysing the solution, it's a single layer convolution being fed to random forrest using flattened arrays
    * Need to take cues from here to maintain pixel intensity value mapping to images
  * Inspired by random forest ML solution, I used bias & dropout in autoencoder model to simulate randomness ....results are visually better but I observe drop in text sharpeness
    * Tried AvgPooling2D for spatial noise correction though am aware it might further worsen text clarity => scrapped(background turned grey against black using MaxPooling2D)
    * Not sure if MaxPooling3D (used to correct temporal & spatial noises together) would help
    * Tried the model on by resizing 258x540 to 420x540 for training/prediction but rmse score hit worst 0.25362 - probably resizing caused intensity variation since for submission I had to resize back to 258x540.

original![test.png](https://raw.githubusercontent.com/kapsdeep/Kaggle-Denoise-Dirty-Documents/master/test.png =260x210) autoencoder![autoencoder.png](https://raw.githubusercontent.com/kapsdeep/Kaggle-Denoise-Dirty-Documents/master/autoencoder_op.png =260x210) median_filter&thresholding![median_filter_&_thesholding](https://raw.githubusercontent.com/kapsdeep/Kaggle-Denoise-Dirty-Documents/master/median_filter_%26_threshold.png =260x210)


## Project Plan
>Lets break the project into independent tasks to enable deeper understanding of individual topics & explore multiple ways to reach the objective

| Project        | ETA          | Risk(s)  | rmse  | Analysis  | Status  |
  | ------------- |:-------------:| -----:| -----:| -----:| -----:|
  | Prj#1: Kaggle competitions using google collab       | 6/16 | None | 0.2 | Worst error without any processing | Done  :white_check_mark: |
| Prj#2: Re-use Prj#1 with "simple background removal model"       | 6/16 | None | 0.06002 | Median filters are used for salt & pepper noise & results point out that a image processing would be helpful for the challenge | Done  :white_check_mark: |
| Prj#3: Build over Prj#2 with "simple linear regression model"       | 6/18 | Other important stuff in weekdays | NA | Problem re-structuring for ML, Scatter plotting training data to arrive at linear relationship, using plt.cmap for image visualisation (didn't find much value pursuing since the expected result is 0.07 rmse) | Dropped:x:     |
| Prj#4: Build over Prj#2 using image processing techniques from top kagglers       | 6/20 | Choose from multiple submissions after reading all solutions | NA | starting with colin's solution: classic ML approach of linear regression=>canny edge detection+GBM=>Adaptive thresholding=>feature enginerring=>stacking=>CNN=>Ensemble. I believe using CNN options that simulate say thresholding will be worhtwile | Dropped:x:     |
| Prj#5: scikit RandomForestRegressor ML model   | 6/22 | can't find equivalent keras functions, trying dropout, bias | 0.02656 | (ML approach using flattened array) Single layer 3x3 convolution(same padding) followed by randomforrestregressor with smart mapping of pixel values to image index | In-progress     |
| Prj#6: Keras implementation using autoencoders       | 6/22 | keeping track of pixel intensity for each image because the CNN works on ndarray & o/p being ndarray I still need to figure out how to tag image to pixel values | 0.05663 | network performs well on background noise. combined with background removal deeper n/w is performing better | In-progress     |
| Prj#7: Implement Prj#6 using winograd & stacking/Ensemble       | 6/25 | need to try out ensemble asap since doing 1st time | NA | NA | Not started     |

#### Top kaggle submissions to read
| Kaggler        | Score          | Kernel  | Packages  |
  | ------------- |:-------------:| -----:| -----:|
  | Alexey Antonov      | 0.02602 | [aantonov Kaggle](https://www.kaggle.com/antonov/kernels) | "as in NN_Starter Kit by Rangel Dokov" numpy as np, theano, theano.tensor as T, cv2, os, itertools, , theano.config.floatX = 'float32'     |
  | Johny Strings      | 0.02602 | [Johnny Strings Kaggle](https://www.kaggle.com/johnnystrings/kernels) | same as Oliver Sherouse     |
  | Colin      | 0.00512 | [Colin Kaggle](https://www.kaggle.com/colinpriest/kernels?sortBy=voteCount&group=everyone&pageSize=20&userId=46478&language=R) | Implemented in R    |
  | Rangel Dokov      | 0.02682 | [Background removal Kaggle](https://www.kaggle.com/rdokov/background-removal/code) | """  Simple background removal code    __author__ : Rangel Dokov    The basic idea is that we have a foreground object of interest (the dark text)  and we want to remove everything that is not part of this foreground object.    This should produce results somewhere around 0.06 on the leaderboard.  """  import numpy as np  from scipy import signal  from PIL import Image |
  | Oliver Sherouse      | 0.02811 | [Let's Try Simple Linear Regression Kaggle](https://www.kaggle.com/oliversherouse/let-s-try-simple-linear-regression/code) | concurrent.futures,csv, logging,random, joblib,numpy as np, sklearn.ensemble, sklearn.cross_validation, sklearn.metrics, skimage.data, from pathlib import Path, from PIL import Image as image     |

#### Prj#1: Kaggle competition using google collab 
1. Enable train/test access in collaboratory
    - (a) Mount google drive to collab
      - [Importing data to Google Colaboratory](https://steemit.com/google-colab/@ankurankan/importing-data-to-google-colaboratory)
      - [Mount gdrive to collab using FUSE](https://colab.research.google.com/drive/1srw_HFWQ2SMgmWIawucXfusGzrj1_U0q)
    - (b) Upload data to google drive & download to collab
        - googledrive APIs
        - Kaggle APIs
          - [Official Kaggle API](https://github.com/Kaggle/kaggle-api)
        - Pydrive (since Byte Order Marker can be removed):heavy_check_mark:
          -  [GitHub & BitBucket HTML Preview](https://htmlpreview.github.io/?https://raw.githubusercontent.com/gsuitedevs/PyDrive/master/docs/build/html/quickstart.html) 
2. Play with image using matplotlib & PIL
    - [Pyplot Image tutorial](https://matplotlib.org/users/image_tutorial.html) :heavy_check_mark:
    - [Tutorial — Pillow (PIL Fork) 5.1.1 documentation](https://pillow.readthedocs.io/en/5.1.x/handbook/tutorial.html):heavy_check_mark:
3. Develop error loss function:interrobang:
    - train & train_cleaned(ground truth) form pair of dataset 
    - test doens't have a ground_truth dataset available hence check rmse check possible after kaggle submission (AFAIK)
4. Develop kaggle submission function
    - download from collab to gdrive:heavy_check_mark:
    - download from collab to localdrive :heavy_check_mark:

#### Prj#2: Re-use Prj#1 with "simple background removal model"
1. Try background removal model to hit rsme: 0.06:heavy_check_mark: 
2. Read through scipy:signal & PIL:Image:heavy_check_mark:
3. Read about Image filtering
    - [umich Image filtering](https://web.eecs.umich.edu/~jjcorso/t/598F14/files/lecture_0924_filtering.pdf)
    - [CS6670: Computer Vision Lecture 2: Image filtering](https://www.cs.cornell.edu/courses/cs6670/2011sp/lectures/lec02_filter.pdf)
    - [CSE.USF Image Filtering](http://www.cse.usf.edu/~r1k/MachineVisionBook/MachineVision.files/MachineVision_Chapter4.pdf)
    - [Auckland Univ: Image Filtering Median Filtering](https://www.cs.auckland.ac.nz/courses/compsci373s1c/PatricesLectures/Image%20Filtering_2up.pdf) :heavy_check_mark: 
    - [Signal Processing (scipy.signal) — Other filters Median Filter](https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html)
    - [Prepress and File Formats: Raster vs Vector](https://www.slideshare.net/JenniferJanviere/prepress)

#### Prj#3: Build over Prj#2 using simple linear regression model
1. Copy flattened arrays of training dataset (train, train_cleaned) :heavy_check_mark:
2. Develop keras linear regression model
3. Apply the model on training dataset
4. Tune hyperparameters

#### Prj#4: Build over Prj#2 using image processing techniques from top kagglers
1. Try adaptive thresholding
2. Try canny edge detection
3. [Gradient Boosting from scratch – ML Review – Medium](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d)
4. [Basics of Ensemble Learning Explained in Simple English](https://www.analyticsvidhya.com/blog/2015/08/introduction-ensemble-learning/)

#### Prj#5: scikit RandomForestRegressor ML model
1. Try the kaggle solution out of box to see it works :heavy_check_mark:
2. Read up scikit functions to understand the model(joblib.Parallel, np.hstack, np.vstack,numpy.ndarray.flatten) :heavy_check_mark:
3. Single layer convolution(as used in get_feature_from_image()) is easy do in keras
4. Find euivalent of random forrest in keras maybe dropout or bias :sos:

#### Prj#6: Keras implementation using autoencoders
1. Compare the model o/p to "simple background removal" :heavy_check_mark:
2. Modify n/w to benefit from other solutions - adaptive thresholding, random forrest, median filter, GBM, Ensemble, Sharpen the text 
3. how to ensure ndarray pixel value corresponds to particlular image in test set ?? :sos:
4. Using deeper Conv layers helped improve text clarity since upsampling was approximating too much("i" was appearing as "l")
    - calculate #layers for input resolution
    - check if different resolutions can be handled at once because 258x540 is predicting black frame for some weird reason
    - trying with resizing 258x540 to 420x540 & training the n/w at once
        - apparently skimage.transform.resize on 258x540 is rendering all pixel intensities as constt value because of which all resized images appear blank in train/test
``` 
TARGET_DIR=Path for i in TARGET_DIR.iterdir(): img = cv2.imread(str(i),0) bottle_resized = resize(img, (420, 540),preserve_range=True) file = str(i).split('/')[4] imsave(file,bottle_resized) </script>
```
      - need to try resize without flattening out pixel value, weird becasue image looks fine hence something to do with numpy array & skimage resize
      - skimage has a bug ![skimage.resize bug with ints · Issue #2702](https://github.com/scikit-image/scikit-image/issues/2702) & workaround is to conver the image to float before resizing & use preserve_range = TRUE
```bottle_resized = resize(img_as_float(x_test[1]), (420, 540),preserve_range=True)```
      - try padding 258x540 to 420x540 - might introduce additional error towards padding
      - median + 10 epochs: score = 0.08923, train_mse = 0.07
median + 50 epochs: score = 0.05663, train_mse = 0.037
median + 100 epochs: score = , train_mse = 0.0017
change np array to float64 & k=3 in medfilt2D
play with kernel size of conv layers

#### Median Filters
>Median filters are non-linear filters used widely for removing salt-pepper noise while preserving edges. Since we need to preseve text while removing noise hence seems to better suited for current problem set. The median is calculated by first sorting all the pixel values from the window into numerical order, and then replacing the pixel being considered with the middle (median) pixel value.

![P6Moj.png](https://i.stack.imgur.com/P6Moj.png =400x210) ![YIhNh.png](https://i.stack.imgur.com/YIhNh.png =400x320)


_Note: TBD_
References
- Papers
    - [Kaggle: Denoising Dirty Documents with MATLAB - File Exchange - MATLAB Central](https://in.mathworks.com/matlabcentral/fileexchange/51812-kaggle--denoising-dirty-documents-with-matlab) (img2csv, submit-developed my own):heavy_check_mark:
    - [Image Processing + Machine Learning in R: Denoising Dirty Documents Tutorial Series No Free Hunch](http://blog.kaggle.com/2015/12/04/image-processing-machine-learning-in-r-denoising-dirty-documents-tutorial-series/)  
    - [Colin Priest](https://colinpriest.com/2015/08/01/denoising-dirty-documents-part-1/)
    - [Denoising with Random Forests - 0.02628](https://www.kaggle.com/johnnystrings/denoising-with-random-forests) 
    - [Denoising stained images using kmeans clustering – Corpocrat Magazine](https://corpocrat.com/2015/07/20/noise-removal-using-kmeans-on-stains-in-scanned-images/)     
    - [Location-aware kernels for large images tiled processing with neural networks](https://medium.com/shallowlearnings/spatial-tiled-processing-for-large-images-with-convolutional-neural-networks-3936ed7aebec) 
    - [(PDF) Enhancement and Cleaning of Handwritten Data by Using Neural Networks](https://www.researchgate.net/publication/221258673_Enhancement_and_Cleaning_of_Handwritten_Data_by_Using_Neural_Networks)
      - [kaggle/denoising-dirty-documents at master · gdb/kaggle · GitHub](https://github.com/gdb/kaggle/tree/master/denoising-dirty-documents)
- Misc
    - [Complete list of github markdown emoji markup · GitHub](https://gist.github.com/rxaviers/7360908)
    - [Noise in photographic images | imatest](http://www.imatest.com/docs/noise/)
    - [5 Tips on Noise Reduction — Cinetic Studios](http://www.cineticstudios.com/blog/2015/12/5-tips-on-noise-reduction.html)
    - [From image files to numpy arrays! | Kaggle](https://www.kaggle.com/lgmoneda/from-image-files-to-numpy-arrays)
    - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
    - [python - Saving a Numpy array as an image - Stack Overflow](https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image)
    - [matplotlib.pyplot.subplots — Matplotlib 2.2.2 documentation](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html)
    - https://archive.ics.uci.edu/ml/datasets/NoisyOffice
- Collab
    - #https://colab.research.google.com/notebooks/io.ipynb#scrollTo=zU5b6dlRwUQk
    - #Alternate: https://github.com/Kaggle/kaggle-api (didn't try kaggle APIs though seemingly easy)
    - #Alternate: https://medium.com/@likho2manish/dev-steps-to-google-colab-5c72779c0ae9 (didn't try googledrive APIs)
    - #Pydrive: https://github.com/gsuitedevs/PyDrive/tree/master/docs 
    - #Pydrive: https://htmlpreview.github.io/?- -https://raw.githubusercontent.com/gsuitedevs/PyDrive/master/docs/build/html/filemanagement.html
    - #Options: https://steemit.com/google-colab/@ankurankan/importing-data-to-google-colaboratory
