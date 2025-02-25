import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        # Load the model from the SavedModel path
        
        
    def predict(self):
        self.model = tf.saved_model.load("C:/Users/Pragyasharava/Desktop/leukemia-detection/artifacts/training/model")
        # Directly load the image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image) / 255.0  # Normalize pixel values to [0, 1]
        test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
        
        # Call the model directly
        result = self.model(test_image)  # Direct call to the model
        
        print("Raw model output:", result)
        
        # Print result for debugging
        print("Model output:", result)
        
        # Assuming the result is a tensor and needs to be converted
        predicted_class = np.argmax(result.numpy(), axis=1)

        # Define class labels
        class_labels = ['[Malignant] early Pre-B', '[Malignant] Pre-B', '[Malignant] Pro-B', 'Benign']
        prediction = class_labels[predicted_class[0]]  # Map to class label
        return [{"image": prediction}]
