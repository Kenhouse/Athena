from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np

def build_model(input_shape):
    model = keras.Sequential([
    layers.Dense(128, activation=tf.nn.relu, input_shape=input_shape),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

def main(unused_argv):
    dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    print(dataset_path)

    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names = column_names, na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

    dataset = raw_dataset.copy()

    #drop the rows with unknown value
    dataset = raw_dataset.dropna()
    #print(dataset)
    #print(dataset.isna().sum())

    #one-hot encoding
    origin = dataset.pop('Origin')
    onehot = pd.get_dummies(origin)
    onehot.columns = ['USA', 'Europe', 'Japan']
    dataset = dataset.join(onehot,lsuffix='MPG', rsuffix='USA')
    print(dataset)

    #split data to train and test
    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    #print('train data: ' + str(len(train_dataset)) + ', test data: ' + str(len(test_dataset)))

    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    #print(train_stats)

    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    #normalize the training&test data
    def norm(x):
        return (x-train_stats['mean'])/train_stats['std']
    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)
    #print(normed_train_data)

    #construct model
    model = build_model([len(train_dataset.keys())])
    #print(model.summary())

    example_batch_data = train_dataset[:10]
    example_batch_result = model.predict(example_batch_data)
    #print(example_batch_result)

    '''Training process'''
    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

    EPOCHS = 1000
    history = model.fit(
      normed_train_data, train_labels,
      epochs=EPOCHS, validation_split = 0.2, verbose=0,
      callbacks=[PrintDot()])
    print('')

    '''print the training history'''
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    #print (hist.tail())

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
    test_predictions = model.predict(normed_test_data).flatten()
    predict = pd.DataFrame({'MPG predict':test_predictions, 'MPG labels':test_labels})
    print(predict.tail())
    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


if __name__ == '__main__':
    tf.app.run()
