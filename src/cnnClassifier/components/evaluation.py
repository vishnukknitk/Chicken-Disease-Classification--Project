import tensorflow as tf
import json
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    def _valid_generator(self):
        
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )
        
        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            interpolation = "bilinear"
        )
        
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory = self.config.training_data,
            subset = "validation",
            shuffle = False,
            **dataflow_kwargs
        )
        
        
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        
        predictions = self.model.predict(self.valid_generator)
        true_labels = self.valid_generator.classes
        
        loss = tf.keras.losses.sparse_categorical_crossentropy(true_labels, predictions).numpy().mean()
        accuracy = tf.keras.metrics.sparse_categorical_accuracy(true_labels, predictions).numpy().mean()

        self.score = [loss, accuracy]
        #self.score = self.model.evaluate(self.valid_generator)
        
    def save_score(self):
        self.scores = {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
        
        # Save the scores to a JSON file
        with open("score.json", "w") as json_file:
            json.dump(self.scores, json_file)
        
        