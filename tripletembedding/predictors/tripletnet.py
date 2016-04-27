import chainer
from chainer import functions as F

from tripletembedding.functions import mse_zero_one
from tripletembedding.functions import sqrt

from chainer import cuda


class TripletNet(chainer.Chain):
    """
    A triplet network remodelling the network proposed in
    Hoffer, E., & Ailon, N. (2014). Deep metric learning using Triplet network.

    The DNN to be used can be passed to the constructor, keeping it inter-
    changeable.
    """

    def __init__(self, dnn):
        super(TripletNet, self).__init__(
            dnn=dnn(),
        )

    def embed(self, x):
        # forward through DNN
        h = self.dnn(x)
        h = F.reshape(h, (h.data.shape[0], h.data.shape[1]))

        # normalize to unit vector
        # h = F.local_response_normalization(h, n=h.data.shape[1]*2,
        #                                    k=0, alpha=1, beta=0.5)
        return h

    def distance(self, anc, pos, neg):
        """
        Compute anchor-positive distance and anchor-negative distance on a
        batch of triplets.
        The batch is forwarded through the network as a whole and then split
        to 3 batches of size n.
        """
        dist_pos = F.expand_dims(F.batch_l2_norm_squared(anc - pos), 1)
        dist_neg = F.expand_dims(F.batch_l2_norm_squared(anc - neg), 1)

        return dist_pos, dist_neg

    def __call__(self, x, margin=0.0, debug=False):
        """
        Forward through DNN and compute loss and accuracy.

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

        # embed
        anc, pos, neg = (self.embed(h) for h in F.split_axis(x, 3, 0))

        dist_pos, dist_neg = self.distance(anc, pos, neg)
        if debug:
            # print("=" * 80)
            # print(dist_pos.data[:5], "\n----\n", dist_neg.data[:5])
            # print("=" * 80)
            import pdb; pdb.set_trace()
        dist = sqrt(F.concat((dist_pos * margin, dist_neg)))

        # compute loss:
        # calculate softmax on distances as a ratio measure
        # loss is MSE of softmax to [0, 1] vector
        sm = F.softmax(dist)
        self.loss = mse_zero_one(sm)

        self.accuracy = (dist_pos.data < dist_neg.data).sum() \
            / len(dist_pos.data)
        self.dist = min(dist_neg.data) - max(dist_pos.data)

        self.nonzero = cuda.cupy.count_nonzero(anc.data) / anc.data.size

        return self.loss
