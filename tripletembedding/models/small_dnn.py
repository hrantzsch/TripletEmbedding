import chainer
import chainer.functions as F
import chainer.links as L


class SmallDnn(chainer.Chain):

    """Adjusted from AlexBN chainer example."""

    def __init__(self):
        super(SmallDnn, self).__init__(
            conv1=L.Convolution2D(1, 32, 10, stride=4, pad=1),
            bn1=L.BatchNormalization(32),
            conv2=L.Convolution2D(32, 64, (1, 2), stride=(1, 2)),
            bn2=L.BatchNormalization(64),
            conv3=L.Convolution2D(64, 64, 5, stride=2, pad=1),
            bn3=L.BatchNormalization(64),
            conv4=L.Convolution2D(64, 32, 3, stride=2, pad=1),
            bn4=L.BatchNormalization(32),
            fc1=L.Linear(128, 128),
            fc2=L.Linear(128, 3)
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x):
        self.clear()

        h = self.conv1(x)
        h = F.max_pooling_2d(h, (3, 5), stride=2)

        h = self.conv2(h)

        h = self.conv3(h)
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)

        h = self.conv4(h)

        h = self.fc1(h)
        h = self.fc2(h)

        return h
