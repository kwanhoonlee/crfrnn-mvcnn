
from __future__ import absolute_import
# from __future__ import division
from __future__ import print_function

from tensorflow import keras
#
# from keras.optimizers import Adam
# from keras.losses import categorical_crossentropy
# from keras.metrics import categorical_accuracy


import inputs
#import pynvml
import argparse

import globals as _g

# _g.set_seed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--use-multi-gpu', action='store_true',
                        help='Using CUDA-enabled device to accelerate training model')
    arg, _ = parser.parse_known_args()

    # define train and validate dataset
    train_dataset, train_steps = inputs.prepare_dataset(_g.TRAIN_LIST)
    print(train_dataset, train_steps)

    val_dataset, val_steps = inputs.prepare_dataset(_g.VAL_LIST)

    # define a MVCNN model
    # model1 = model._cnn1((277,277,3))
    # _g.VIEWS_IMAGE_SHAPE
    import sys

    sys.path.insert(1, './src')
    import model
    # model1 = model.inference_multi_view_without_crfrnn()
    model1 = model.inference_multi_view_with_crfrnn()
    model1.summary()

    # origin_model = model
    model1.compile(optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-5),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])

    print("dataset ",train_dataset)
    model1.fit(train_dataset, epochs=_g.NUM_TRAIN_EPOCH, steps_per_epoch=train_steps,
              validation_data=val_dataset, validation_steps=val_steps)
    # start training model
    # train_dataset.repeat(1)
    #
    # graph_model.fit(train_dataset.repeat(1), epochs=10, steps_per_epoch=1, batch_size=None,
    #           validation_data=val_dataset, validation_steps=val_steps)
    # model.fit(train_dataset, epochs=_g.NUM_TRAIN_EPOCH, steps_per_epoch=train_steps)
    # save model's wights
    # origin_model.save_weights('model/latest.weights.h5', save_format='h5')
