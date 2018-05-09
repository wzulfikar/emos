import sys
import argparse
import tensorflow as tf
from mfcc import mfcc
from halo import Halo

def serving_input_receiver_fn():
  """Build the serving inputs."""
  # The outer dimension (None) allows us to batch up inputs for
  # efficiency. However, it also means that if we want a prediction
  # for a single instance, we'll need to wrap it in an outer list.
  inputs = {
    "MFCC1": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC2": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC3": tf.placeholder(shape=[None, 15], dtype=tf.float32),
    "MFCC4": tf.placeholder(s`hape=[None, 15], dtype=tf.float32),
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

def train(train_file: str, test_file: str, batch_size: int, steps: int):
    # Fetch the data
    (train_x, train_y), (test_x, test_y) = mfcc.load_data(train_file, test_file)

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        model_dir="model_directory",

        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        })

    # Train the Model.
    classifier.train(
        input_fn=lambda:mfcc.train_input_fn(train_x, train_y, batch_size),
        steps=steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:mfcc.eval_input_fn(test_x, test_y, batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    '''
    MODIFY THE CODE STARTING FROM HERE.
    YOU CAN:
    1) Change the filename to what audio file to extract the feature (mfcc in this case)
    '''

    # how many loops. in future iteration, change into infinite loop
    for x in range(0,1):
        '''
        #this is the live stream audio coding
        stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
        print("start....")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("done...")


        stream.stop_stream()
        stream.close()
        #p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        '''
    
    # export_dir = classifier.export_savedmodel(
    #     export_dir_base="export_here",
    #     serving_input_receiver_fn=serving_input_receiver_fn)
    
def my_model(features, labels, mode, params):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
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
    with Halo(text='training data..'):
        train(train_file=args.train_file, 
              test_file=args.test_file,
              batch_size=args.batch_size, 
              steps=args.train_steps)
