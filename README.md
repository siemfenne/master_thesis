# Master Thesis

## Dependencies
The CNNs used for the MNIST classification task and CBIS-DDSM classification task are built using Google's Tensorflow 2.0 API Keras. It is therefore very importart (apart from the usual packages) to correctly install Tensorflow. I believe (not sure though) that the installment on windows/linux is pretty straightforward. However, this study was conducted using a macbook pro with M1 chip. Unfortunately, Tensorflow is not yet straightaway compatible with the new M1/2/3 chips of apple. Therefore, one has to go through a manual installation procedure, creating environments, installing certain versions of packages, etc. To get it to work while using GPU instead of CPU, I have followed the following tutorial: https://www.youtube.com/watch?v=5DgWvU0p2bk . The tutorial requires to specify the versions of the depencies. This can be found in the "tensorflow-gpu.yml" file.

## Data
The CBIS-DDSM dataset was downloaded from: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset .  However, a clean unaltered version can be found in the "CBIS-DDSM dataset clean" directory. If one wants to include the exploratory data analysis or the data preprocessing of the project, this clean version has to be moved and renamed to another directory. This will be explained below.

## Notebooks
The project was done using 5 jupyter notebooks and 1 normal script, each using Python. To run everything successfully, a certain order of running the notebooks has to be maintained. Here below, we give this order while describing what eachs notebook/script does:

### 1. eda.ipynb
Exploratory data analysis (EDA) notebook. Here we explored the CBIS-DDSM dataset. Some of the figures in the thesis in the section "Exploratory data analysis" can be found here. Since the EDA was performend on unaltered image data, to run this notebook, one needs to use the unaltered CBIS-DDSM version mentioned before. Follow the following steps:
1. Move "CBIS-DDSM dataset clean" to the "Data" directory.
2. Rename the "CBIS-DDSM dataset" directory in "Data" to "something else" (this is the altered version)
3. Rename the "CBIS-DDSM dataset clean" directory in "Data" to "CBIS-DDSM dataset"
4. Now you can run it

Please note that this notebook was run at the very beginning of this study, and it might be possible that some code gives errors due to changed directory names or other things.

### 2. montecarlo.ipynb
Notebook in which the monte carlo simulation was conducted. Nothing needed to run this notebook except a working version of Tensorflow. (I load the MNIST dataset with Tensorflow)

### 3. mnist.ipynb
Notebook in which I conducted the MNIST classification task for the logistic regression and simple CNN. It also shows the visualization analysis conducted in the thesis. Nothing needed to run this notebook except a working version of Tensorflow.

### 4. dataprep.py
Python script with all the functions that conduct the data preprocessing script. These functions are later imported in the notebook that conducts the CBIS-DDSM classification.

### 5. cbis-cnn.ipynb
Notebook in which the CBIS-DDSM classification was done with ResNet50 plus the visualization analysis. If you want to run the data preprocessing as well, we again need to move and rename some of the directories. We distinguish two cases:
Case 1: You DID NOT run the eda.ipynb notebook and the directories are still named and placed at the original location. Follow these steps:
1. Follow the exact same steps described in the "1. eda.ipynb" section here above
2. Now you can run it

Case 2: You did DID run the eda.ipynb notebook and changed/relocated directories, follow these steps:
1. You can run it

If you do not want to run the data preprocessing (hence you run the notebook from section "modeling"), again 2 cases:
Case 1: You DID NOT run the eda.ipynb notebook, follow these steps:
1. You can run it

Case 2: You DID run the eda.ipynb notebook, follow these steps:
1. Change the name of "CBIS-DDSM dataset" back to "CBIS-DDSM dataset clean"
2. Change the name of the "something else" directory back to "CBIS-DDSM dataset"

### 6. cbis-lasso.ipynb
Notebook in which the CBIS-DDSM classification was done using the lasso regularized logistic regression. IMPORTANT: this notebook assumes you have done the data preprocessing steps. Hence you can run it if in "Data/CBIS-DDSM dataset" has the following directory: base_dir, csv, masked, png. If that is the case, you did the preprocessing and you can continue running this notebook.

## Figures
In this directory, all the figures (and possibly more) from the thesis can be found. Note that these can also be found in the notebooks themself. They are showed underneath each cell.

## Models_mnist / Models_unfrozen
Directories where I saved the CNN models for the MNIST and CBIS-DDSM classification tasks.

