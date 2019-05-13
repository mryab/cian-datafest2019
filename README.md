# cian-datafest2019
2nd place solution for cian.ru Data Fest 2019 competition

## How to run

Download competition data and reproduce the folder structure used in Kaggle kernels. 
Specifically, images must be in `input/train/train` and `input/test/test` folders respectively

Then run the script to obtain `submission.csv` in src folder:
```
cd src
python3 -O fit_predict.py
```

## Description

As the competition constraints were to use only numpy, scipy and matplotlib, I decided to use a simple neural network with 3 convolutional and 2 dense layers. The code uses the numpy CNN implementation from [simple-neural-networks](https://github.com/MorvanZhou/simple-neural-networks) repo, so most of the credit goes to the author of this library. 

I added the dropout layer to prevent possible overfitting and trained the network on 75x75 center crops of normalized images with Adam optimizer. The code should produce a submission file in ~30 minutes while using about 9GB of RAM.