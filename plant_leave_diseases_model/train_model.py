import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from plant_leave_diseases_model.config.core import config
from plant_leave_diseases_model.model import classifier
from plant_leave_diseases_model.processing.data_manager import load_train_dataset, load_validation_dataset, load_test_dataset, callbacks_and_save_model,prepare_img_data
from plant_leave_diseases_model.processing.data_setup import test_directory,val_directory,train_directory

from sklearn.preprocessing import LabelBinarizer
import cv2
from keras.preprocessing.image import img_to_array, array_to_img
import numpy as np
from os import listdir
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import pickle

default_image_size = tuple((256, 256))
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
    
def run_training() -> None:
    
    """
    Train the model.
    """
    #x,y=prepare_img_data(train_directory)
    directory_root =train_directory
    image_list, label_list = [], []
    try:
        print("[INFO] Loading images ...")
        root_dir = listdir(directory_root)
        
        for plant_folder in root_dir :
            if not plant_folder.startswith('.DS_Store'):
                plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
                print(f"{directory_root}/{plant_folder}")
                for plant_disease_folder in plant_disease_folder_list:
                    if not plant_disease_folder.startswith('.DS_Store'): 
                        image_directory = (f"{directory_root}/{plant_folder}/{plant_disease_folder}")
                        if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                            image_list.append(convert_image_to_array(image_directory))
                            label_list.append(plant_folder)
        print("[INFO] Image loading completed")  
    except Exception as e:
        print(f"Error : {e}")
        
    image_size = len(image_list)
    print("image_size:",image_size)
    print("label_list:",label_list)
    print("len(image_list):",len(image_list))
    
    
    

   
    label_binarizer = LabelBinarizer()
    image_labels = label_binarizer.fit_transform(np.array(label_list))
    pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
    n_classes = len(label_binarizer.classes_)
    print("n_classes::",n_classes)
    print("image_labels:",image_labels)
    print("image_labels:",np.array(image_labels))
    
    
    np_image_list = np.array(image_list, dtype=np.float16) / 255.0
    
    print("np_image_list.shape:",np_image_list.shape)
    print("image_labels.shape:",image_labels.shape)

    '''
    x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 
    
    
    print("x_train:shape::",x_train.shape)
    print("y_train:shape::",y_train.shape)
    #for i in y:
    #    print("i:",i)
        
    #for i in x:
    #    print("xi:",i)  
        
    #classifier.fit()
    
    #print("x_train:",x_train.shape,"::length:",len(x_train))
    #$print("y_train:",y_train.shape,"::length:",len(y_train))
    
    
    history = classifier.fit(
    np_image_list, encoded_arr, batch_size=5,
    #validation_data=(x_test, y_test),
    #steps_per_epoch=len(x_train) // 5,
    epochs=1
    )
    '''

          
    
    #train_data = load_train_dataset()
    #val_data = load_validation_dataset()
    #test_data = load_test_dataset()

    '''
    # Model fitting
    classifier.fit(train_data,
                   epochs = config.model_config.epochs,
                   validation_data = val_data,
                   callbacks = callbacks_and_save_model(),
                   verbose = config.model_config.verbose
                   )
    '''
    # Calculate the score/error
    #test_loss, test_acc = classifier.evaluate(test_data)
    #print("Loss:", test_loss)
    #print("Accuracy:", test_acc)
    
    
if __name__ == "__main__":
    run_training()