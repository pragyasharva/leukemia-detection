import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        
        
    def predict(self):
        self.model = load_model("artifacts/training/model.keras")
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image) / 255.0  # Normalize pixel values to [0, 1]
        test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
        
        result = self.model(test_image)  # Direct call to the model
        
        print("Raw model output:", result)
        
        # Print result for debugging
        print("Model output:", result)
        
        # Assuming the result is a tensor and needs to be converted
        predicted_class = np.argmax(result.numpy(), axis=1)

        # Define class labels
        class_labels = ['Benign', '[Malignant] Pre-B', '[Malignant] Pro-B', '[Malignant] early Pre-B']
        prediction = class_labels[predicted_class[0]]  # Map to class label
        return [{"image": prediction}]
