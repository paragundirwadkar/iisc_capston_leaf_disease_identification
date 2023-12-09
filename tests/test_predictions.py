"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
from plant_leave_diseases_model import __version__ as _version
from plant_leave_diseases_model.config.core import config
from plant_leave_diseases_model.predict import load_model_and_predict, make_prediction
from plant_leave_diseases_model.processing.data_manager import get_model_file_name_path, load_model


def test_make_prediction(sample_input_data):
    # Given
    data, labels = sample_input_data
    data_in = data[0]
    print("test_make_prediction:data_in:",data_in)

    # When
    results = load_model_and_predict(data_in)
    
    y_pred = results[0]
    print()
    # Then
    assert y_pred is not None
    #assert y_pred in ['cat', 'dog']
    assert results['version'] == _version


def test_accuracy(sample_input_data):
    # Given
    data, labels = sample_input_data
    
    # When
    model_file_name = get_model_file_name_path()
    model = load_model(file_name = model_file_name)
    test_loss, test_acc = model.evaluate(data, labels, verbose=0)
    
    print("test_make_prediction:test_accuracy:(test_loss,test_acc):",test_loss,",",test_acc)
    # Then:
    assert test_acc > 0.0
