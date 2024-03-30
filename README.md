# Master Thesis
Going through all the notebooks, you can see how all the results and figures of the thesis were computed. All results and figures are also still displayed in the notebooks themself under each cell with code corresponding to that result/figure. If one wants to run the code, please read the instructions below:

## Dependencies
The CNNs used for the MNIST classification task and CBIS-DDSM classification task are built using Google's Tensorflow 2.0 API Keras. It is therefore very importart (apart from the usual packages) to correctly install Tensorflow. The installment on windows/linux is pretty straightforward and can be done using pip, see: https://www.tensorflow.org/install/pip. However, this study was conducted using a macbook pro with M1 chip. Unfortunately, Tensorflow is not yet straightaway compatible with the new M1/2/3 chips of apple. Therefore, one has to go through a manual installation procedure, creating environments, installing certain versions of packages, etc. To get it to work while using GPU instead of CPU, I have followed the following tutorial: https://www.youtube.com/watch?v=5DgWvU0p2bk . The tutorial requires to specify the versions of the depencies. This can be found in the "tensorflow-gpu.yml" file.

## Data
Since the CBIS-DDSM dataset has size 6GB, we have not included it in this GitHub repository. Therefore, the data can be downloaded from: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset . After downloading, please rename the directory to "CBIS-DDSM dataset" and place it in another directory called "Data" such that you have the following directory structure:

Data/
    CBIS-DDSM dataset/

This is crucial for the notebooks to work.

## Notebooks
The project was done using 5 jupyter notebooks and 1 normal script, each using Python. To run everything successfully, a certain order of running the notebooks has to be maintained. Here below, we give this order while describing what eachs notebook/script does:

### 1. eda.ipynb
Exploratory data analysis (EDA) notebook. Here we explored the CBIS-DDSM dataset. Some of the figures in the thesis in the section "Exploratory data analysis" can be found here. Since the EDA was performed at the very early stages of the project, we recognize two possible issues when running this notebook:
1. Due to name changes of files or other changes in the code, running this notebook might cause errors
2. If one still tries to run it, to make sure the other notebooks run smoothly, please remove the data, download it again, and locate it with the same names in the same directory structure as mentioned above.

Unfortunately, there was no time left to change the notebook such that everything runs smoothly. However, some of the figures of the EDA seen in the thesis can be found in the notebook.

### 2. montecarlo.ipynb
Notebook in which the monte carlo simulation was conducted. Nothing needed to run this notebook except a working version of Tensorflow. (I load the MNIST dataset with Tensorflow)

### 3. mnist.ipynb
Notebook in which I conducted the MNIST classification task for the logistic regression and simple CNN. It also shows the visualization analysis conducted in the thesis. Nothing needed to run this notebook except a working version of Tensorflow.

### 4. dataprep.py
Python script with all the functions that conduct the data preprocessing script. These functions are later imported in the notebook that conducts the CBIS-DDSM classification.

### 5. cbis-cnn.ipynb
Notebook in which the CBIS-DDSM classification was done with ResNet50 plus the visualization analysis. Make sure you have a correct data directory structure.

### 6. cbis-lasso.ipynb
Notebook in which the CBIS-DDSM classification was done using the lasso regularized logistic regression. IMPORTANT: this notebook assumes you have done the data preprocessing steps in the "cbis-cnn.ipynb" notebook. Hence you can run it if in "Data/CBIS-DDSM dataset" has the following directories: base_dir, csv, masked, png. If that is the case, you did the preprocessing and you can continue running this notebook.

## Figures
In this directory, all the figures (and possibly more) from the thesis can be found. Note that these can also be found in the notebooks themself. They are showed underneath each cell.

## Models_mnist / Models_unfrozen
Directories where I saved the CNN models for the MNIST and CBIS-DDSM classification tasks.

