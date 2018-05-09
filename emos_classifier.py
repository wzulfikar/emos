import numpy as np
import tensorflow as tf
from mfcc import loader
import scipy.io.wavfile as wav
from python_speech_features import mfcc

class Classifier:
    def __init__(self,
                model_dir: str = './model_directory', 
                train_file: str = './mfcc/mfcc_test.csv', 
                test_file: str = './mfcc/mfcc_training.csv'):
      config = tf.estimator.RunConfig(model_dir=model_dir)
      
      (train_x, train_y), (test_x, test_y) = loader.load_data(train_file, test_file)

      # Feature columns describe how to use the input.
      my_feature_columns = []
      for key in train_x.keys():
          my_feature_columns.append(tf.feature_column.numeric_column(key=key))

      self.feature_columns = my_feature_columns
      self.train_x = train_x
      self.train_y = train_y

      self.test_x = test_x
      self.test_y = test_y
    
      self.keys = [
            'MFCC1',
            'MFCC2',
            'MFCC3',
            'MFCC4',
            'MFCC5',
            'MFCC6',
            'MFCC7',
            'MFCC8',
            'MFCC9',
            'MFCC10',
            'MFCC11',
            'MFCC12',
            'MFCC13',
            'MFCC14',
            'MFCC15'
        ]

      self.classifier = tf.estimator.Estimator(
          config=config,
          model_dir=model_dir,
          model_fn=self._model,
          params={
              'feature_columns': my_feature_columns,
              # Two hidden layers of 10 nodes each.
              'hidden_units': [10, 10, 10],
              # The model must choose between 3 classes.
              'n_classes': 3,
          })

    def _model(self, features, labels, mode, params):
        """DNN with three hidden layers, and dropout of 0.1 probability."""
        # Create three fully connected layers each layer having a dropout
        # probability of 0.1.
        net = tf.feature_column.input_layer(features, params['feature_columns'])
        for units in params['hidden_units']:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

        # Compute logits (1 per class).
        logits = tf.layers.dense(net, params['n_classes'], activation=None)

        # Compute predictions.
        predicted_classes = tf.argmax(logits, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.softmax(logits),
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Compute loss.
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Compute evaluation metrics.
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predicted_classes,
                                       name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        # Create training op.
        assert mode == tf.estimator.ModeKeys.TRAIN

        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def train(self, batch_size: int, steps: int):
        # Train the Model.
        self.classifier.train(
            input_fn=lambda:loader.train_input_fn(self.train_x, self.train_y, batch_size),
            steps=steps)

        # Evaluate the model.
        eval_result = self.classifier.evaluate(
            input_fn=lambda:loader.eval_input_fn(self.test_x, self.test_y, batch_size))

        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    def predict(self, wavfile: str, batch_size) -> [(str, float)]:
        (rate, sig) = wav.read(wavfile)
        mfcc_feat = mfcc(sig, rate, numcep=15)
        b=np.mean(mfcc_feat, axis=0)
        array=[[i] for i in b]

        predict_x= dict(zip(self.keys, array))

        # print('predict x', predict_x)
        predictions = self.classifier.predict(
            input_fn=lambda:loader.eval_input_fn(predict_x,
                                                    labels=None,
                                                    batch_size=batch_size))
        
        predictions_out = []
        for pred_dict in predictions:
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]

            predictions_out.append((loader.SPECIES[class_id], probability))

        return predictions_out
