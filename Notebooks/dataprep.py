# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import seaborn as sns
import shutil
import cv2
import random
from sklearn.model_selection import train_test_split

def load_dataframes(get_unique_images:bool, drop_without_callback:bool, total:bool):
    """
    Puts all the file names of the image into a dataframe.

    get_unique_image: only include the unique directory names
    drop_without_callback: pathology includes benign, malignant, benign_without_callback. If True, latter is dropped
    total: if True, returns all images into one dataframe
    """
    df_mass_train = pd.read_csv('../Data/CBIS-DDSM dataset/csv/mass_case_description_train_set.csv')
    df_mass_test = pd.read_csv('../Data/CBIS-DDSM dataset/csv/mass_case_description_test_set.csv')
    df_calc_train = pd.read_csv('../Data/CBIS-DDSM dataset/csv/calc_case_description_train_set.csv')
    df_calc_test = pd.read_csv('../Data/CBIS-DDSM dataset/csv/calc_case_description_test_set.csv')

    if get_unique_images:
        # get unique images
        df_calc_train = df_calc_train.drop_duplicates(subset='cropped image file path', keep='last').reset_index(drop=True)
        df_calc_test = df_calc_test.drop_duplicates(subset='cropped image file path', keep='last').reset_index(drop=True)
        df_mass_train = df_mass_train.drop_duplicates(subset='cropped image file path', keep='last').reset_index(drop=True)
        df_mass_test = df_mass_test.drop_duplicates(subset='cropped image file path', keep='last').reset_index(drop=True)

    if drop_without_callback:
        # drop benign without callback
        df_calc_train = df_calc_train[df_calc_train['pathology'] != 'BENIGN_WITHOUT_CALLBACK'].reset_index(drop=True)
        df_calc_test = df_calc_test[df_calc_test['pathology'] != 'BENIGN_WITHOUT_CALLBACK'].reset_index(drop=True)
        df_mass_train = df_mass_train[df_mass_train['pathology'] != 'BENIGN_WITHOUT_CALLBACK'].reset_index(drop=True)
        df_mass_test = df_mass_test[df_mass_test['pathology'] != 'BENIGN_WITHOUT_CALLBACK'].reset_index(drop=True)

    if total:
        return pd.concat([df_calc_train, df_calc_test]).reset_index(drop=True), pd.concat([df_mass_test, df_mass_train]).reset_index(drop=True)
    else:      
        return df_mass_train, df_mass_test, df_calc_train, df_calc_test
    
def move_masked_images_from_cropped_directories(df, threshold):

    """
    Removes masked ROI images from cropped images directories

    df: takes dataframe from previous function as input
    threshold: threshold for deciding whether image in directory is a masked or cropped one
    """

    print('######################################')
    print('#                                    #')
    print('#      Preproccesing started         #')
    print('#                                    #')
    print('######################################')


    print('------------------ Removing masked images ------------------')

    os.makedirs(f"../Data/CBIS-DDSM dataset/masked", exist_ok=True)

    for i in range(len(df)):
        img_dir_path = df['cropped image file path'][i].split('/')[2]

        image_names = os.listdir(f'../Data/CBIS-DDSM dataset/jpeg/{img_dir_path}')

        for image_name in image_names:
            source_path = f'../Data/CBIS-DDSM dataset/jpeg/{img_dir_path}/{image_name}'
            img = Image.open(source_path)
            img = img.resize((227, 227), Image.LANCZOS)

            pixel_values = list(img.getdata())
            zero_pixel = 1
            nonzero_pixel = 1
            for pixel in pixel_values:
                if pixel == 0:
                    zero_pixel += 1
                else:
                    nonzero_pixel += 1

            pixel_ratio = nonzero_pixel / zero_pixel

            if pixel_ratio < threshold:
                destination_path = f'../Data/CBIS-DDSM dataset/masked/{image_name}'
                shutil.move(source_path, destination_path)
    
    print('Successfully completed')

def convert_images_to_png(df):

    """
    Converts images from jpg to png format (png format is a lossless format, better for data preprocessing)
    """

    print('---------------- Converting images to PNG ----------------')

    for i in range(len(df)):
        img_dir_path = df['cropped image file path'][i].split('/')[2]
        image_names = os.listdir(f'../Data/CBIS-DDSM dataset/jpeg/{img_dir_path}')

        for image_name in image_names:
            source_path = f'../Data/CBIS-DDSM dataset/jpeg/{img_dir_path}/{image_name}'
            image = Image.open(source_path)
            png_path = os.path.join(f'../Data/CBIS-DDSM dataset/jpeg/{img_dir_path}', image_name.rsplit('.', 1)[0] + '.png')
            image.save(png_path, 'PNG')

            
            os.remove(source_path)

    os.rename('../Data/CBIS-DDSM dataset/jpeg', '../Data/CBIS-DDSM dataset/png')

    print('Successfully completed')

def image_augmentation(df, rotation_angles):

    """
    Applies image augmentation

    df: takes in dataframe with all image names
    rotation_angles: list of rotational angles
    """

    print('------------------ Applying augmentation ------------------')

    for i in range(len(df)):
        img_dir_path = df['cropped image file path'][i].split('/')[2]
        image_names = os.listdir(f'../Data/CBIS-DDSM dataset/png/{img_dir_path}')

        for image_name in image_names:
            source_path = f'../Data/CBIS-DDSM dataset/png/{img_dir_path}/{image_name}'
            image = Image.open(source_path)
            for angle in rotation_angles:
                rotated_image = image.rotate(angle, expand=True)
                rotated_image.save(f'../Data/CBIS-DDSM dataset/png/{img_dir_path}/{image_name}_{angle}.png')

    print('Successfully completed')

def apply_clahe_on_images(df: pd.DataFrame, clip_limit: int):

    """
    Applies CLAHE image enhancement technique

    df: takes in dataframe with all image names
    clip_limit: defines the clip level for enhancing
    """

    print('------------------ Applying CLAHE ------------------')

    for i in range(len(df)):
        img_dir_path = df['cropped image file path'][i].split("/")[2]
        image_names = os.listdir(f'../Data/CBIS-DDSM dataset/png/{img_dir_path}')

        for image_name in image_names:
            source_path = f'../Data/CBIS-DDSM dataset/png/{img_dir_path}/{image_name}'
            image = Image.open(source_path)
            image_np = np.array(image)
            height, width = image_np.shape[:2]
            tile_size = (
                int(np.ceil(height / 80)),
                int(np.ceil(width / 80))
            )

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            # clahe = cv2.createCLAHE()
            clahe_image = clahe.apply(image_np)
            cv2.imwrite(source_path, clahe_image)
    
    print('Successfully completed')

def image_resizing(df:pd.DataFrame, resizing_dims:tuple):
    
    """
    Resizes the images to required fixed size 224x224 

    df: takes in dataframe with image names
    resizing_dims: tuple with the 224x224 dimensions
    """

    print('------------------ Resizing images ------------------')

    resizing_dimensions = resizing_dims[:2]

    for i in range(len(df)):
        img_dir_path = df['cropped image file path'][i].split("/")[2]
        image_names = os.listdir(f'../Data/CBIS-DDSM dataset/png/{img_dir_path}')

        for image_name in image_names:
            source_path = f'../Data/CBIS-DDSM dataset/png/{img_dir_path}/{image_name}'
            image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_resized = cv2.resize(image_rgb, resizing_dimensions, cv2.INTER_AREA)
            cv2.imwrite(source_path, image_resized)

    print('Successfully completed')

def split_data(df, train_size=0.8, val_size=0.1, test_size=0.1, random_seed=42):
    
    """
    Randomly splits data into 80% train, 10% validation, 10% test

    random_seed: used to have the same random split over multiple iterations
    """

    print('------------------ Creating train, test, and validation sets ------------------')

    df_malignant = df[df['pathology'] == 'MALIGNANT'].reset_index(drop=True)
    df_benign = df[df['pathology'].isin(['BENIGN', 'BENIGN_WITHOUT_CALLBACK'])].reset_index(drop=True)

    df_train_malignant, df_remaining_malignant = train_test_split(df_malignant, train_size=train_size, random_state=random_seed)
    df_train_benign, df_remaining_benign = train_test_split(df_benign, train_size=train_size, random_state=random_seed)

    relative_val_size = val_size / (val_size + test_size)

    # Split the remaining data into validation and test sets
    df_val_malignant, df_test_malignant = train_test_split(df_remaining_malignant, train_size=relative_val_size, random_state=random_seed)
    df_val_benign, df_test_benign = train_test_split(df_remaining_benign, train_size=relative_val_size, random_state=random_seed)

    print('Successfully completed')

    return df_train_malignant.reset_index(drop=True), df_train_benign.reset_index(drop=True), df_val_malignant.reset_index(drop=True), df_val_benign.reset_index(drop=True), df_test_malignant.reset_index(drop=True), df_test_benign.reset_index(drop=True)

def safe_move(src, dst):
    """
    Some images have the same name while different patient. If that is the case, alter the name of the image

    src: source directory
    dst: destination directory
    """
    if not os.path.exists(src):
        print(f"File not found: {src}")
        return
    if not os.path.exists(dst):
        shutil.move(src, dst)
    else:
        base, extension = os.path.splitext(dst)
        i = 1
        new_dst = f"{base}_{i}{extension}"
        while os.path.exists(new_dst):
            i += 1
            new_dst = f"{base}_{i}{extension}"
        shutil.move(src, new_dst)

def create_train_val_test_dirs(df_train_malignant, df_train_benign, df_val_malignant, df_val_benign, df_test_malignant, df_test_benign):

    """
    Creates train, validation, test directory structure that is 
    compatible with Tensorflow's ImageDataGenerator function
    """

    print('------------------ Moving images to directory structure for training ------------------')

    train_dir = '../Data/CBIS-DDSM dataset/base_dir/train/'
    val_dir = '../Data/CBIS-DDSM dataset/base_dir/validation/'
    test_dir = '../Data/CBIS-DDSM dataset/base_dir/test/'

    for dir in ['benign', 'malignant']:
        os.makedirs(train_dir+dir, exist_ok=True)
        os.makedirs(val_dir+dir, exist_ok=True)
        os.makedirs(test_dir+dir, exist_ok=True)
    
    for i in range(len(df_train_malignant)):
        img_dir_path = df_train_malignant['cropped image file path'][i].split("/")[2]
        image_names = os.listdir(f'../Data/CBIS-DDSM dataset/png/{img_dir_path}')
        for image_name in image_names:
            source_path = f'../Data/CBIS-DDSM dataset/png/{img_dir_path}/{image_name}'
            destination_path = os.path.join(train_dir, 'malignant', image_name)
            safe_move(source_path, destination_path)

    for i in range(len(df_train_benign)):
        img_dir_path = df_train_benign['cropped image file path'][i].split("/")[2]
        image_names = os.listdir(f'../Data/CBIS-DDSM dataset/png/{img_dir_path}')
        for image_name in image_names:
            source_path = f'../Data/CBIS-DDSM dataset/png/{img_dir_path}/{image_name}'
            destination_path = os.path.join(train_dir, 'benign', image_name)
            safe_move(source_path, destination_path)

    for i in range(len(df_val_malignant)):
        img_dir_path = df_val_malignant['cropped image file path'][i].split("/")[2]
        image_names = os.listdir(f'../Data/CBIS-DDSM dataset/png/{img_dir_path}')
        for image_name in image_names:
            source_path = f'../Data/CBIS-DDSM dataset/png/{img_dir_path}/{image_name}'
            destination_path = os.path.join(val_dir, 'malignant', image_name)
            safe_move(source_path, destination_path)

    for i in range(len(df_val_benign)):
        img_dir_path = df_val_benign['cropped image file path'][i].split("/")[2]
        image_names = os.listdir(f'../Data/CBIS-DDSM dataset/png/{img_dir_path}')
        for image_name in image_names:
            source_path = f'../Data/CBIS-DDSM dataset/png/{img_dir_path}/{image_name}'
            destination_path = os.path.join(val_dir, 'benign', image_name)
            safe_move(source_path, destination_path)

    for i in range(len(df_test_malignant)):
        img_dir_path = df_test_malignant['cropped image file path'][i].split("/")[2]
        image_names = os.listdir(f'../Data/CBIS-DDSM dataset/png/{img_dir_path}')
        for image_name in image_names:
            source_path = f'../Data/CBIS-DDSM dataset/png/{img_dir_path}/{image_name}'
            destination_path = os.path.join(test_dir, 'malignant', image_name)
            safe_move(source_path, destination_path)

    for i in range(len(df_test_benign)):
        img_dir_path = df_test_benign['cropped image file path'][i].split("/")[2]
        image_names = os.listdir(f'../Data/CBIS-DDSM dataset/png/{img_dir_path}')
        for image_name in image_names:
            source_path = f'../Data/CBIS-DDSM dataset/png/{img_dir_path}/{image_name}'
            destination_path = os.path.join(test_dir, 'benign', image_name)
            safe_move(source_path, destination_path)

    print('Successfully completed')

    print('######################################')
    print('#                                    #')
    print('#      Preproccesing ended           #')
    print('#                                    #')
    print('######################################')

def print_class_distribution(source_dir, set):

    """
    Prints class distribution after data preprocessing
    """

    print(f"------------- {set} set  --------------")
    cases_b = len(os.listdir(f"{source_dir}/benign"))
    print(f"Benign cases: {cases_b}")
    cases_m = len(os.listdir(f"{source_dir}/malignant"))
    print(f"Malignant cases: {cases_m}")
    total_cases_train = cases_b + cases_m
    print(f"Total cases: {total_cases_train}")
