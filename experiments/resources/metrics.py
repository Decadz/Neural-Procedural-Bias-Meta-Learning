import torch


class ErrorRate:

    def __call__(self, pred, true):
        correct, incorrect = 0, 0

        for fx, y in zip(pred, true):
            if y == torch.argmax(fx):
                correct += 1
            else:
                incorrect += 1

        return torch.tensor(1 - (correct / (correct + incorrect)))


# Note, if you use this you need to update checkpointers to maximizing.
class Accuracy:

    def __call__(self, pred, true):
        correct, incorrect = 0, 0

        for fx, y in zip(pred, true):
            if y == torch.argmax(fx):
                correct += 1
            else:
                incorrect += 1

        return torch.tensor(correct / (correct + incorrect))
