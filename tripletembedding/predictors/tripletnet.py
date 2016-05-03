import chainer
from chainer import functions as F

from tripletembedding.functions import mse_zero_one
from tripletembedding.functions import sqrt

from chainer import cuda

class TripletNet(chainer.Chain):
    """
    A triplet network remodelling the network proposed in
    Hoffer, E. & Ailon, N. (2014). Deep metric learning using Triplet network.

    The CNN to be used is supplied as an argument to the constructor.
    """

    def __init__(self, cnn):
        super(TripletNet, self).__init__(
            cnn=cnn(),
        )

    def _accuracy(self, dist_pos, dist_neg):
        """
        Calculate share of samples where anc-pos distance is smaller than
        anc-neg distance.
        """
        return (dist_pos.data < dist_neg.data).sum() / dist_pos.data.size

    def _max_distance(self, dist_pos, dist_neg):
        """
        Calculate difference between the shortest anc-neg distance and the
        largest anc-pos distance; i.e. the worst embedding in this batch.
        Larger is better.
        """
        return min(dist_neg.data) - max(dist_pos.data)

    def embed(self, x):
        """Forward through the CNN"""
        h = self.cnn(x)
        h = F.reshape(h, (h.data.shape[0], h.data.shape[1]))
        return h

    def squared_distance(self, anc, pos, neg):
        """
        Compute anchor-positive distance and anchor-negative distance on
        batches of anchors, positive, and negative samples.
        """
        dist_pos = abs(anc - pos)  # TODO betrag!
        dist_neg = abs(anc - neg)

        return dist_pos, dist_neg

    def compute_loss(self, dist_pos, dist_neg, margin_factor=1.0, dims=3):
        """
        Use Softmax on the distances as a ratio measure and compare it to a
        vector of [[0, 0, ...] [1, 1, ...]] (Mean Squared Error).
        This function also computes the accuracy and the 'max_distance'.
        """
        # apply margin factor and take square root
        # dist = sqrt(F.concat((dist_pos * margin_factor, dist_neg)))

        dist_pos_split = F.split_axis(dist_pos, dist_pos.data.shape[1], 1)
        dist_neg_split = F.split_axis(dist_neg, dist_neg.data.shape[1], 1)

        splices = (F.concat((dist_pos_split[i], dist_neg_split[i])) for i in range(len(dist_pos_split)))

        softmaxes = F.concat(F.softmax(s) for s in splices)

        n = len(dist_pos.data)
        zero_one = cuda.cupy.array([0, 1] * dims, dtype=cuda.cupy.float32)
        zero_one = chainer.Variable(cuda.cupy.vstack([zero_one] * n))
        zero_one.unchain_backward()

        self.loss = F.mean_squared_error(softmaxes, zero_one)
        self.accuracy = self._accuracy(dist_pos, dist_neg)
        # self.dist = self._max_distance(dist_pos, dist_neg)
        self.dist = 0

        return self.loss

    def __call__(self, x, margin_factor=1.0, debug=False):
        """
        Embed samples using the CNN, then calculate distances and triplet loss.

        x is a batch of size 3n following the form:

        | anchor_1   |
        | [...]      |
        | anchor_n   |
        | positive_1 |
        | [...]      |
        | positive_n |
        | negative_1 |
        | [...]      |
        | negative_n |
        """
        anc, pos, neg = (self.embed(h) for h in F.split_axis(x, 3, 0))
        dist_pos, dist_neg = self.squared_distance(anc, pos, neg)
        return self.compute_loss(dist_pos, dist_neg, margin_factor)
