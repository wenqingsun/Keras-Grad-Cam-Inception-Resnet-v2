from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.applications.inception_resnet_v2 import preprocess_input
from sklearn import metrics
import numpy as np
import os

batch_size = 32
save_dir = '../models/'
experiment_name = 'baseline_try2'
model_name = 'trained_model_inception-resnetv2_tl_baseline.h5'

model = load_model(os.path.join(save_dir, experiment_name, model_name))

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


valid_generator = val_datagen.flow_from_directory(
    directory=r"../data/val/",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
    seed=42
)




# # Score trained model.
# loss = model.evaluate_generator(generator=valid_generator)
# print('Test loss:', loss)

scores = model.predict_generator(generator=valid_generator)

correct = 0
for i, n in enumerate(valid_generator.filenames):
    if n.startswith("benign") and scores[i][0] > 0.5:
        correct += 1
    if n.startswith("malignant") and scores[i][0] <= 0.5:
        correct += 1

print("Correct:", correct, " Total: ", len(valid_generator.filenames))
#print("Loss: ", loss, "Accuracy: ", correct/len(valid_generator.filenames))

fpr, tpr, thresholds = metrics.roc_curve(valid_generator.classes, scores[:,1], pos_label=1)
auc = metrics.auc(fpr, tpr)
print('AUC is ', auc)

print('This is experiment ', experiment_name)