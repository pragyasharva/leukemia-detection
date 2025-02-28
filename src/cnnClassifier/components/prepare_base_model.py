import os
from pathlib import Path
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        # Ensure image_size is a list
        image_size = list(self.config.params_image_size)


        weights = self.config.params_weights
        if isinstance(weights, tf.Tensor):
            weights = weights.numpy().decode('utf-8')  

        # Print the values to verify
        print("Image Size:", image_size)
        print("Weights:", weights)

        # Create the model
        self.model = tf.keras.applications.EfficientNetB3(
            input_shape=image_size,
            weights=weights,  
            include_top=self.config.params_include_top
        )



        # Save the model 
        self.model.save(self.config.base_model_path) 





    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False


        # Add custom classification layers
        flatten_in = GlobalAveragePooling2D()(model.output)
        dense_1 = Dense(128, activation='relu')(flatten_in)
        dropout = Dropout(0.6)(dense_1)  # Increased Dropout (More Regularization)
        prediction = Dense(classes, activation="softmax")(dropout)
        

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        # Compile the model
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        # Save the model in TensorFlow SavedModel format
        self.full_model.save(self.config.updated_base_model_path)

    
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

        

