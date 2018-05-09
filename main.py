import sys
import argparse
import tensorflow as tf
from halo import Halo
import emos_classifier 

def serving_input_receiver_fn():
  """Build the serving inputs."""
  # The outer dimension (None) allows us to batch up inputs for
  # efficiency. However, it also means that if we want a prediction
  # for a single instance, we'll need to wrap it in an outer list.
  inputs = {
    "MFCC1": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC2": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC3": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC4": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC5": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC6": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC7": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC8": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC9": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC10": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC11": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC12": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC13": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC14": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC15": tf.placeholder(shape=[None, 15], dtype=tf.float32),
  }
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', type=str, help='wav file to predict')
    parser.add_argument('--train', default=False, type=bool, help='train the model')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--train_steps', default=1000, type=int,
                        help='number of training steps')
    parser.add_argument('--train_file', 
                        default="./mfcc/mfcc_training.csv", 
                        type=str,
                        help='mfcc training file (csv)')
    parser.add_argument('--test_file', 
                        default="./mfcc/mfcc_training.csv", 
                        type=str,
                        help='mfcc test file (csv)')

    args = parser.parse_args(sys.argv[1:])

    print('- loading classifier..')
    classifier = emos_classifier.Classifier()
    
    if args.predict:
        print('- predicting file:', args.predict)
        predictions = classifier.predict(args.predict, args.batch_size)
        print('predictions:', predictions)

    if args.train:
        with Halo(text='training data..'):
            classifier.train(batch_size=args.batch_size, 
                            steps=args.train_steps)
