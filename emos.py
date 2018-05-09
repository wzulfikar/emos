from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from tensorflow.python.estimator.estimator import Estimator

KEYS = [
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

def predict(wavfile: str, classifier: Estimator, mfcc_data, batch_size) -> (str, float):

    filename= (wavfile)
    (rate,sig) = wav.read(filename)
    mfcc_feat = mfcc(sig,rate,numcep=15)
    b=np.mean(mfcc_feat, axis=0)
    array=[[i] for i in b]

    predict_x= dict(zip(KEYS, array))

    print(predict_x)
    predictions = classifier.predict(
        input_fn=lambda:mfcc_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=batch_size))
    print(predictions)

    for pred_dict in predictions:
        template = ('\nPrediction is "{}" ({:.1f}%)')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(mfcc_data.SPECIES[class_id],
                              100 * probability))
