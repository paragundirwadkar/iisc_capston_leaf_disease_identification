import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from plant_leave_diseases_model.config.core import config,PACKAGE_ROOT,TRAINED_MODEL_DIR
from plant_leave_diseases_model.model import create_model
from plant_leave_diseases_model.processing.data_manager import get_one_hot_data_for_input_classes, load_leaf_disease_dataset, load_train_dataset, load_validation_dataset, load_test_dataset, callbacks_and_save_model,prepare_img_data,get_class_file_list,get_model_file_name_path,get_master_classes_in_data_frame
from plant_leave_diseases_model.processing.data_setup import load_dataset_images, load_leaf_classes, prepare_data_images_per_class, print_dir, test_directory,val_directory,train_directory,class_file_path

from sklearn.preprocessing import LabelBinarizer
import cv2
from keras.preprocessing.image import img_to_array, array_to_img
import numpy as np
from os import listdir
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import pickle
from sklearn.preprocessing import OneHotEncoder
from plant_leave_diseases_model import __version__ as _version 
#from plant_leave_diseases_model.predict import make_prediction

default_image_size = tuple((256, 256))
    
def run_training() -> None:
    
    """
    Train the model.
    """
    
    ##########################################
    # Get train data : x_train, y_train
    ##########################################
    x_train, y_train,leaf_disease_master_classes = load_leaf_disease_dataset(train_directory)

    print("x_train_size:",len(x_train))
    print("y_train_size:",len(y_train))
    print("x_train.shape:",x_train.shape)
    print("y_train.shape:",y_train.shape)
    
    ##########################################
    # Get validation data : x_val, y_val
    ##########################################
    x_val, y_val,leaf_disease_master_classes = load_leaf_disease_dataset(val_directory)

    print("x_val_size:",len(x_val))
    print("y_val_size:",len(y_val))
    print("x_val.shape:",x_val.shape)
    print("y_val.shape:",y_val.shape)
    
    ##############################################################
    # Getting no of classes to pass thw model at last layer
    ##############################################################
    n_classes = len(leaf_disease_master_classes)
    
    ################################
    # Create model
    ################################
    model = create_model(input_shape = config.model_config.input_shape, 
                          optimizer = config.model_config.optimizer, 
                          loss = config.model_config.loss, 
                          metrics = [config.model_config.accuracy_metric],
                          n_classes = n_classes
                        )
    ################################
    # Training the model
    ################################
  
    history = model.fit(
    x_train, 
    y_train, 
    batch_size=config.model_config.batch_size,
    validation_data=(x_val, y_val),
    #steps_per_epoch=len(x_train) // 5,
    #callbacks = callbacks_and_save_model(),
    epochs=config.model_config.epochs
    )

    ################################
    # Saving the model
    ################################
    save_model_file_name = get_model_file_name_path()
    print("###################### ####### ##########################")
    print("save_model_file_name:",save_model_file_name)
    print("###################### ####### ##########################")

    model.save(save_model_file_name)
    
if __name__ == "__main__":
    print_dir()
    load_dataset_images()
    load_leaf_classes()
    prepare_data_images_per_class()
    run_training()
    # Define directory where test images are loaded
    print("###################### ####### ##########################")
    print("###################### PREDICT ##########################")
    print("###################### ####### ##########################")

    #test_dir_img_file_path=test_directory+"/Apple___Apple_scab"
    #print("test_dir_img_file_path::",test_dir_img_file_path) 
    #make_prediction(test_dir_img_file_path = test_dir_img_file_path)