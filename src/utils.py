#Saving the  preprocessor object to a file using pickle
import os
import sys
import pickle
from src.logger import logger
from src.exception import CustomException   
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True) # Create the directory if it doesn't exist

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj) # Save the object to the specified file path
    except Exception as e:
        raise CustomException(e, sys)