from __future__ import print_function
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers
from keras import metrics

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
import os

def plot_history(history, history_dir):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    xc = range(epochs)
    box = dict(facecolor='yellow', pad=5, alpha=0.2)
    fig, axes = plt.subplots(nrows=2, ncols=2)  # create figure & 1 axis
    fig.subplots_adjust(left=0.2, wspace=0.6)
    axes[0, 0].plot(xc, train_loss)
    axes[0, 0].set_title('training')
    axes[0, 0].set_ylabel('loss', bbox=box)
    axes[0, 1].plot(xc, val_loss)
    axes[0, 1].set_title('validation')
    axes[0, 1].set_ylabel('loss', bbox=box)
    axes[1, 0].plot(xc, train_acc)
    axes[1, 0].set_ylabel('accuracy', bbox=box)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 1].plot(xc, val_acc)
    axes[1, 1].set_ylabel('accuracy', bbox=box)
    axes[1, 1].set_ylim(0, 1)
    fig.savefig(history_dir)  # save the figure to file
    plt.close(fig)  # close the figure

def train_model():
    # load stage 1 model
    base_model_path = os.path.join(save_dir, experiment_name, base_model_name)
    model = load_model(base_model_path)
    print("Loaded model from disk")


    # train all layers
    for layer in model.layers:
        layer.trainable = True

    # compile the model (should be done *after* setting layers to non-trainable)
    adam = optimizers.Adam(lr=learning_rate, decay=decay)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])

    train_datagen  = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        preprocessing_function = preprocess_input) # use inception-resnet-v2 preprocessing func

    val_datagen  = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        preprocessing_function = preprocess_input) # use inception-resnet-v2 preprocessing func

    train_generator = train_datagen.flow_from_directory(
        directory=r"../data/train/",
        target_size=(299, 299),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42)

    valid_generator = val_datagen.flow_from_directory(
        directory=r"../data/val/",
        target_size=(299, 299),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=42
    )

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    history = model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=epochs)

    model.evaluate_generator(generator=valid_generator)


    # Save model and weights
    if not os.path.isdir(os.path.join(save_dir, experiment_name)):
        os.makedirs(os.path.join(save_dir, experiment_name))
    model_path = os.path.join(save_dir, experiment_name, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # save log history
    # visualizing losses and accuracy
    history_dir = os.path.join(save_dir, experiment_name, model_name + '_history_2.jpg')
    plot_history(history, history_dir)

#=====================================================================================================
batch_size = 32
num_classes = 2
epochs = 30
learning_rate = 0.0005
decay = 0.7
save_dir = '../models/'
experiment_name = 'two_stage'
base_model_name = 'trained_model_inception-resnetv2_tl_two_stage_1.h5'
model_name = 'trained_model_inception-resnetv2_tl_two_stage_2.h5'

if __name__ == '__main__':
    train_model()
