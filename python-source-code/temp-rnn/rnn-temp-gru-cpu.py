import os
import numpy as np

# To disable all logging output from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(2)

import matplotlib.pyplot as plt
import pandas as pd

import tensorflow.keras as keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

with tf.device('/cpu:0'):
    # dataset training, testing, validation generator
    def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):

        if max_index is None:
            max_index = len(data) - delay - 1

        i = min_index + lookback

        while 1 :

            if shuffle:
                # it returns a batch_size of random rows
                rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                # it returns a batch_size of rows in sequence
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)

            # batch_size x (lookback // step) x data.shape[-1]
            samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
            targets = np.zeros((len(rows),))

            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j] + delay][1]

            yield samples, targets



    def evaluate_naive_method(val_steps, val_gen):
        batch_maes = []
        for step in range(val_steps):
            if (step % 1000 == 0):
                print("We got to the ", step, " step.")
            samples, targets = next(val_gen)
            preds = samples[:, -1, 1]
            mae = np.mean(np.abs(preds - targets))
            batch_maes.append(mae)
        return np.mean(batch_maes)


    def make_model(float_data):
        # Create a new linear regression model.
        model = Sequential()
        model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
        model.add(layers.Dense(1))
        model.compile(optimizer=RMSprop(), loss='mae')
        return model


    # Prepare a directory to store all the checkpoints.
    checkpoint_dir = './ckpt-gru'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    def make_or_restore_model(float_data):
        # Either restore the latest model, or create a fresh one
        # if there is no checkpoint available.
        checkpoints = [checkpoint_dir + '/' + name
                    for name in os.listdir(checkpoint_dir)]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print('Restoring from', latest_checkpoint)
            return keras.models.load_model(latest_checkpoint)
        print('Creating a new model')
        return make_model(float_data)


    print( " " )
    print( "Read the jena_climate_2009_2016.csv" )
    print( " " )
    data_dir = 'jena_climate'
    fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
    f = open(fname)
    data = f.read()
    f.close()


    print( "-----------------------------------" )


    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    print( "Dataset header:" )
    print(header)
    print( " " )
    print( "Number of dataset lines:" )
    print(len(lines))
    print( " " )


    print( "-----------------------------------" )


    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values
    print( "Dataset shape:" )
    print( float_data.shape )
    print( " " )
    print( "Dataset first 2 lines" )
    print( float_data[0:2] )
    print( " " )

    print( "-----------------------------------" )

    print( "Normalizing training data" )
    print( " " )
    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std
    print( "Training data first 2 lines" )
    print( float_data[0:2] )
    print( " " )

    print( "-----------------------------------" )

    print( "Instantiate three generators: one for training, one for validation, and one for testing")
    print( " " )
    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128

    train_gen = generator(float_data,
        lookback=lookback,
        delay=delay,
        min_index=0,
        max_index=200000,
        shuffle=True,
        step=step,
        batch_size=batch_size)

    val_gen = generator(float_data,
        lookback=lookback,
        delay=delay,
        min_index=200001,
        max_index=300000,
        step=step,
        batch_size=batch_size)

    test_gen = generator(float_data,
        lookback=lookback,
        delay=delay,
        min_index=300001,
        max_index=None,
        step=step,
        batch_size=batch_size)

    print( "-----------------------------------" )

    print( "Dataset lines used for training")
    print ("0-200000")
    print( "Dataset lines used for validation")
    print ("200001-300000")
    print( "Dataset lines used for testing")
    print ("300001-420550")
    print( " " )

    print( "-----------------------------------" )

    val_steps = (300000 - 200001 - lookback)
    test_steps = (len(float_data) - 300001 - lookback)
    print( "How many steps to draw from val_gen in order to see the entire validation set")
    print( val_steps)
    print( "How many steps to draw from test_gen in order to see the entire test set")
    print( test_steps )

    print( "-----------------------------------" )

    print( "Computing the common-sense baseline mean absolute error (MAE)" )
    # mae = evaluate_naive_method(val_steps, val_gen)
    # print( mae )
    # celsius_mae = mae * std[1]
    print( 0.29 )
    celsius_mae = 0.29 * std[1]
    print( "Convert MAE to Celsius error" )
    print( celsius_mae )

    print( "-----------------------------------" )

    # Fitting RNN to training set using Keras Callbacks. Read Keras callbacks docs for more info.
    regressor = make_or_restore_model(float_data)
    print(regressor.summary())

    print( "-----------------------------------" )

    print("Model input shapes")
    for layer in regressor.layers:
        print(layer.input_shape)

    print( "-----------------------------------" )

    print( "Train model" )

    callbacks = [
        # This callback saves a SavedModel every 100 epochs.
        # We include the training accuracy in the folder name.
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + '/ckpt-ep{epoch:03d}-valloss{val_loss:.4f}',
            period='epoch',
            monitor='val_loss')
    ]

    set_epochs = 0
    history = regressor.fit(train_gen,
        epochs=set_epochs,
        steps_per_epoch=500,
        validation_data=val_gen,
        validation_steps=val_steps,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1)

    print( "-----------------------------------" )

    if (set_epochs != 0 ):
        print( "Plot loss" )

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

    print( "-----------------------------------" )

    number_of_batches_to_test = 1
    print( "Plot prediction result for ", number_of_batches_to_test," number of samples" )
    print( "RNN takes as input (samples, timesteps, features)" )
    print( "We trained the network to take as input (samples, timesteps, features) = (samples, 240, 14)" )
    print( "So, one RNN output will be valid after 240 timesteps" )
    print( "In our RNN, the first layer is a GRU feeding a dense layer." )
    print( "The GRU layer will be valid for the dense layer after 240 timesteps. ")


    # get the test samples and feed them as input to the regressor
    # get the test targets (groundtruth) to compare them to the regressor results
    test_samples = []
    test_targets = []
    j = 0
    for i in test_gen:
        if ( j >= (number_of_batches_to_test) ):
            break
        else:
            # i is equal to the (samples, targets) of the generator
            test_samples.append(i[0])
            test_targets.append(i[1])
        j=j+1
    test_samples = np.asarray(test_samples)
    test_targets = np.asarray(test_targets)

    print("The shape of the test_samples is: ",test_samples.shape)
    print("The shape of the test_targets is: ",test_targets.shape)

    print("We reshape the samples and the targets to be ready for prediction and plot.")
    test_samples = test_samples.reshape( j*batch_size, 240, 14 )
    test_targets = test_targets.reshape( j*batch_size, 1 )
    print("The shape of the test_samples is: ",test_samples.shape)
    print("The shape of the test_targets is: ",test_targets.shape)

    print("Perform prediction")
    predicted_temp = regressor.predict(test_samples, verbose=1, use_multiprocessing=True)

    print("Predicted values shape and length:")
    print(np.asarray(predicted_temp).shape)
    print(len(predicted_temp))
    predicted_targets = predicted_temp.copy()

    print("De-normalize predicted and groundtruth (test_targets) values")
    test_targets *= std[1]
    test_targets += mean[1]
    predicted_temp *= std[1]
    predicted_temp += mean[1]

    print("mean =", mean[1])
    print("std =", std[1])

    plt.figure()
    plt.title('Real temperature and predicted')
    plt.plot( range( 0,len(test_targets) ), predicted_temp, label = "predicted")
    plt.plot( range( 0,len(test_targets) ), test_targets, label = "real")
    # show a legend on the plot
    plt.xlabel("hours")
    plt.ylabel("celcius")

    # absolute difference
    abs_diff = abs(test_targets - predicted_temp)

    # mean absolute difference
    mean_abs_diff_a = abs_diff.mean(axis=0)
    mean_abs_diff = [mean_abs_diff_a] * len(test_targets)
    plt.annotate(str(round(mean_abs_diff_a[0], 2)), xy=(0,mean_abs_diff_a[0]))
    plt.plot( range( 0,len(test_targets) ), mean_abs_diff, label = "mean absolute difference")

    # mean squared error
    squared_diff = np.square(test_targets - predicted_temp)
    sum_squared_diff = np.sum(squared_diff)
    mean_squared_a = sum_squared_diff/len(test_targets)
    mean_squared = [mean_squared_a] * len(test_targets)
    plt.annotate(str(round(mean_squared_a, 2)), xy=(0,mean_squared_a))
    plt.plot( range( 0,len(test_targets) ), mean_squared, label = "MSE")


    plt.legend()

    plt.show()

    print( "-----------------------------------" )

    print( "Extracting the weights" )

    units = int(int(regressor.layers[0].trainable_weights[0].shape[1])/3)
    print("Number of units: ", units)

    GRU_layer = regressor.layers[0]
    W = GRU_layer.get_weights()[0]
    U = GRU_layer.get_weights()[1]
    b = GRU_layer.get_weights()[2]

    W_z = W[:,          : units    ]
    W_r = W[:, units    : units * 2]
    W_h = W[:, units * 2:          ]

    U_z = U[:,          : units    ]
    U_r = U[:, units    : units * 2]
    U_h = U[:, units * 2:          ]

    b_xz = b[0,          : units    ]
    b_xr = b[0, units    : units * 2]
    b_xh = b[0, units * 2:          ]
    b_hz = b[1,          : units    ]
    b_hr = b[1, units    : units * 2]
    b_hh = b[1, units * 2:          ]

    print("W parameter count = ", W_z.size+W_r.size+W_h.size)
    print("U parameter count = ", U_z.size+U_r.size+U_h.size)
    print("b parameter count = ", b_xz.size+b_xr.size+b_xh.size+b_hz.size+b_hr.size+b_hh.size)
    print("GRU parameter count = ", W.size+U.size+b.size)

    np.savetxt('./parameters-gru/W_z.csv', W_z, delimiter=',')
    np.savetxt('./parameters-gru/W_r.csv', W_r, delimiter=',')
    np.savetxt('./parameters-gru/W_h.csv', W_h, delimiter=',')

    np.savetxt('./parameters-gru/U_z.csv', U_z, delimiter=',')
    np.savetxt('./parameters-gru/U_r.csv', U_r, delimiter=',')
    np.savetxt('./parameters-gru/U_h.csv', U_h, delimiter=',')

    np.savetxt('./parameters-gru/b_xz.csv', b_xz, delimiter=',')
    np.savetxt('./parameters-gru/b_xr.csv', b_xr, delimiter=',')
    np.savetxt('./parameters-gru/b_xh.csv', b_xh, delimiter=',')
    np.savetxt('./parameters-gru/b_hz.csv', b_hz, delimiter=',')
    np.savetxt('./parameters-gru/b_hr.csv', b_hr, delimiter=',')
    np.savetxt('./parameters-gru/b_hh.csv', b_hh, delimiter=',')

    dense_layer = regressor.layers[1]
    W_dense = dense_layer.get_weights()[0]
    b_dense = dense_layer.get_weights()[1]

    print("W_dense parameter count = ", W_dense.size)
    print("b_dense parameter count = ", b_dense.size)

    np.savetxt('./parameters-gru/W_dense.csv', W_dense, delimiter=',')
    np.savetxt('./parameters-gru/b_dense.csv', b_dense, delimiter=',')


    test_samples = test_samples.reshape( j*batch_size*240*14,1 )
    #test_targets = test_targets.reshape( j*batch_size, 1 )

    print("Test samples count = ", test_samples.size)
    print("Test targets count = ", test_targets.size)
    print("Predicted targets count = ", predicted_targets.size)

    np.savetxt('./parameters-gru/test_samples_normalized.csv', test_samples, delimiter=',')
    np.savetxt('./parameters-gru/test_targets_denormalized.csv', test_targets, delimiter=',')
    np.savetxt('./parameters-gru/predicted_targets_normalized.csv', predicted_targets, delimiter=',')


    test_set = float_data[300001:]
    print("test set length: ", len(test_set))
    test_set = test_set.reshape( len(test_set)*14 ,1)

    np.savetxt('./parameters-gru/test_sets_normalized.csv', test_set, delimiter=',')