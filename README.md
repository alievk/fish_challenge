# N+1 fish, N+2 fish challenge

This repo contains my custom bounding circles detector developed for the challenge. It's made on TensorFlow.

The detector is based on the SqueezeDet architecture. The base SqueezeNet network is replaced with ResNet50. The output feature map is then convolved with 3x3 kernel and output features are {confidence, class probabilities, cx, cy, r} x k, where cx, cy, r are center of the circle and its radius. k is the number of anchor circles. I used anchor 5 circles with radii computed as k-means of the train objects radii. The loss function contains three parts. First is a standard cross-entropy loss for classes. Second is confidence loss computed basically as mean (IOU - confidence)^2, where IOU is intersection over union of predicted and ground truth bounding circles. The third part is SmoothL1 of circle parameters divergency (L2 leads to gradient explosion).

For details, referto these sources:
src/nets/resnet50_convDet.py - detector
src/nn_skeleton.py - detector interpretation, loss, train graphs.
src/fish_db.py - data preparation
src/train.py - network training
