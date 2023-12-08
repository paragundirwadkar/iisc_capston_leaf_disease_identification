import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import typing as t
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from keras.utils import image_dataset_from_directory
from plant_leave_diseases_model.config.core import config
from plant_leave_diseases_model import __version__ as _version
from plant_leave_diseases_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from plant_leave_diseases_model.processing.data_setup import load_dataset_images,load_leaf_classes,test_directory,val_directory,train_directory

import cv2
from keras.preprocessing.image import img_to_array, array_to_img
import numpy as np
from os import listdir

def download_images():
    load_dataset_images()

def load_leaf_classes_from_dataset():
    return load_leaf_classes()
    
def load_train_dataset():
    train_dataset = image_dataset_from_directory(directory = train_directory,
                                                image_size = config.model_config.image_size,
                                                batch_size = config.model_config.batch_size)    
    return train_dataset


def load_validation_dataset():
    validation_dataset = image_dataset_from_directory(directory = val_directory,
                                                    image_size = config.model_config.image_size,
                                                    batch_size = config.model_config.batch_size)
    return validation_dataset


def load_test_dataset():
    test_dataset = image_dataset_from_directory(directory = test_directory,
                                                image_size = config.model_config.image_size,
                                                batch_size = config.model_config.batch_size)
    return test_dataset


# Define a function to return a commmonly used callback_list
def callbacks_and_save_model():
    callback_list = []
    
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.model_save_file}{_version}.keras"
    save_path = TRAINED_MODEL_DIR / save_file_name
    print("save_path::",save_path)
    save_path = "model_save.keras"

    #remove_old_model(files_to_keep = [save_file_name])

    # Default callback
    callback_list.append(keras.callbacks.ModelCheckpoint(filepath = save_path,
                                                         #save_best_only = config.model_config.save_best_only,
                                                         monitor = config.model_config.monitor))

    if config.model_config.earlystop > 0:
        callback_list.append(keras.callbacks.EarlyStopping(patience = config.model_config.earlystop))

    return callback_list


def load_model(*, file_name: str) -> keras.models.Model:
    """Load a persisted model."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = keras.models.load_model(filepath = file_path)
    return trained_model


def remove_old_model(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old models.
    This is to ensure there is a simple one-to-one mapping between the package version and 
    the model version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

#Converting Images to array
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, (256,256))
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

image_list, label_list = [], []
    
def prepare_img_data(directory_root):
    
    try:
        print("[INFO] Loading images ...")
        root_dir = listdir(directory_root)

        for plant_folder in root_dir :
            plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
            print(f"{directory_root}/{plant_folder}")
            for plant_disease_folder in plant_disease_folder_list:
                image_directory = (f"{directory_root}/{plant_folder}/{plant_disease_folder}")
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    label_list.append(plant_folder)
        print("[INFO] Image loading completed")
        return image_list,label_list
    except Exception as e:
        print(f"Error : {e}")