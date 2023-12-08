import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from plant_leave_diseases_model.config.core import config,PACKAGE_ROOT,TRAINED_MODEL_DIR
from plant_leave_diseases_model.model import create_model
from plant_leave_diseases_model.processing.data_manager import load_train_dataset, load_validation_dataset, load_test_dataset, callbacks_and_save_model,prepare_img_data
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
    #x,y=prepare_img_data(train_directory)
    directory_root =val_directory
    image_list, label_list = [], []
    
    directory_root =val_directory
    image_list, label_list = prepare_img_data(directory_root)
        
    image_size = len(image_list)
    print("image_size:",image_size)
    print("label_list:",label_list)
    print("len(image_list):",len(image_list))
    
    #e_dataframe = pd.DataFrame(label_list)  
    n_classes=5

    '''
    label_binarizer = LabelBinarizer()
    image_labels = label_binarizer.fit_transform(label_list)
    pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
    n_classes = len(label_binarizer.classes_)
    print("n_classes::",n_classes)
    print("image_labels:",image_labels)
    print("image_labels:",np.array(image_labels))
    '''
    
    np_image_list = np.array(image_list, dtype=np.float16) / 255.0
    
    print("np_image_list.shape:",np_image_list.shape)
    #print("image_labels.shape:",image_labels.shape)
    
   
    text_file = open(class_file_path, "r")
    img_classes = text_file.readlines()
    print (img_classes)
    print ("length:",len(img_classes),"::img_classes:",img_classes[0])
    text_file.close()
   
    leaf_disease_classes_from_input = pd.DataFrame(label_list,columns=['class'])
    print("leaf_disease_classes_from_input:",leaf_disease_classes_from_input)
    
    img_classes = config.model_config.leaf_class_master_category_list 
    leaf_disease_master_classes = pd.DataFrame(img_classes,columns=['class'])
    print("leaf_disease_master_classes:",leaf_disease_master_classes)
    
    
    
    ohe = OneHotEncoder()
    ohe.fit(leaf_disease_master_classes[['class']])
    transformed = ohe.transform(leaf_disease_classes_from_input[['class']])
    #transformed = ohe.fit_transform(e_dataframe[['class']])
    print(transformed.toarray())
    print("len of final classes:",len(transformed.toarray()))
    #print(transformed)
    
    n_classes = len(leaf_disease_master_classes)
    
    print("n_classes::",n_classes)
    print("config.model_config.input_shape::",tuple(config.model_config.input_shape))
    # Create model
    model = create_model(input_shape = config.model_config.input_shape, 
                          optimizer = config.model_config.optimizer, 
                          loss = config.model_config.loss, 
                          metrics = [config.model_config.accuracy_metric],
                          n_classes = n_classes
                        )


    history = model.fit(
    np_image_list, transformed.toarray(), batch_size=5,
    #validation_data=(x_test, y_test),
    #steps_per_epoch=len(x_train) // 5,
    #callbacks = callbacks_and_save_model(),
    epochs=1
    )
    
    save_file_name = f"{config.app_config.model_save_file}{_version}.keras"
    model.save(str(TRAINED_MODEL_DIR)+"/"+str(save_file_name))
          
    
    
    
if __name__ == "__main__":
    run_training()