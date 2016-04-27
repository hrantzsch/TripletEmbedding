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
            fc2=L.Linear(128, 32)
        )
        self.train = True
        self.relu = False

    def clear(self):
        self.loss = None
        self.accuracy = None

    def maybe_relu(self, x):
        if self.relu:
            return F.relu(x)
        else:
            return x

    def __call__(self, x):
        self.clear()

        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(self.maybe_relu(h), (3, 5), stride=2)

        h = self.bn2(self.conv2(h), test=not self.train)
        h = self.maybe_relu(h)

        h = self.bn3(self.conv3(h), test=not self.train)
        h = F.max_pooling_2d(self.maybe_relu(h), 3, stride=2, pad=1)

        h = self.bn4(self.conv4(h), test=not self.train)
        h = self.maybe_relu(h)

        h = self.fc1(h)
        h = self.fc2(h)

        return h
