import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from plant_leave_diseases_model.config.core import config,PACKAGE_ROOT,TRAINED_MODEL_DIR
from plant_leave_diseases_model.model import create_model
from plant_leave_diseases_model.processing.data_manager import get_one_hot_data_for_input_classes, load_train_dataset, load_validation_dataset, load_test_dataset, callbacks_and_save_model,prepare_img_data,get_class_file_list,get_model_file_name_path,get_master_classes_in_data_frame
from plant_leave_diseases_model.processing.data_setup import test_directory,val_directory,train_directory,class_file_path

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

default_image_size = tuple((256, 256))
    
def run_training() -> None:
    
    """
    Train the model.
    """
   
    ##############################
    # Get validation image set
    ##############################
    train_image_list, train_label_list = prepare_img_data(val_directory)
    
    ##############################
    # Get validation image set
    ##############################
    val_image_list, val_label_list = prepare_img_data(train_directory)  
    
    print("train_image_list_size:",len(train_image_list))
    print("train_label_list_size:",len(train_label_list))
    print("train_image_list.shape:",train_image_list.shape)

    print("val_image_list_size:",len(val_image_list))
    print("val_label_list_size:",len(val_label_list))
    print("val_image_list.shape:",val_image_list.shape)
       
    ##########################################
    # Get master class data in data frames
    ##########################################
    leaf_disease_master_classes = get_master_classes_in_data_frame()    
    print("leaf_disease_master_classes:",leaf_disease_master_classes)

    ##########################################
    # Get one hot encoded train labels
    ##########################################
    y_train = get_one_hot_data_for_input_classes(leaf_disease_master_classes, train_label_list)
  
    print("len of final y_train classes:",len(y_train))
    
    
    ##########################################
    # Get one hot encoded validation labels
    ##########################################
    y_val = get_one_hot_data_for_input_classes(leaf_disease_master_classes, val_label_list)
        
    print("len of final y_val classes:",len(y_val))
        
    ##############################################################
    # Getting no of classes to pass thw model at last layer
    ##############################################################
    n_classes = len(leaf_disease_master_classes)
    
    print("n_classes::",n_classes)
    #print("config.model_config.input_shape::",tuple(config.model_config.input_shape))
    
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
    x_train=train_image_list
    y_train=y_train
    x_val=val_image_list
    
    history = model.fit(
    x_train, 
    y_train, 
    batch_size=config.model_config.batch_size,
    validation_data=(x_val, y_val),
    #steps_per_epoch=len(x_train) // 5,
    #callbacks = callbacks_and_save_model(),
    epochs=1
    )

    ################################
    # Saving the model
    ################################
    save_file_name = get_model_file_name_path()
    model.save(str(TRAINED_MODEL_DIR)+"/"+str(save_file_name))
    
if __name__ == "__main__":
    run_training()