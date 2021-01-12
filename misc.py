import tensorflow as tf


class LRScheduleWithReduceOnPlato:
    def __init__(self, init_lr, patience, factor):
        self.lr = init_lr
        self.last_loss = float("inf")
        self.counter = 0
        self.patience = patience
        self.factor = factor

    def get_lr(self):
        return self.lr

    def update_loss(self, cur_loss):
        if cur_loss >= self.last_loss:
            self.counter += 1
            if self.counter == self.patience:
                self.counter = 0
                self.lr = self.lr * self.factor
        else:
            self.counter = 0
            self.last_loss = cur_loss

    @staticmethod
    def round(x, decimals=0):
        multiplier = tf.constant(10 ** decimals, dtype=tf.float32)
        return tf.round(x * multiplier) / multiplier
