## Grad-CAM implementation in Keras ##
Part of the code borrowed from https://github.com/jacobgil/keras-grad-cam

Able to run in keras 2.0

Part 1: (transfer learning folder)
To train a transfer learning model:
run train_inception_resnetv2.py

Or use two stage training:
run train_inception_resnetv2_two_stage_1.py
then run run train_inception_resnetv2_two_stage_2.py

Part 2: (grad-cam_for_trained_model folder)
To get the grad cam results:
run grad-cam-inception_resnetv2-batch-requested_v2.py (the other two versions also work)

