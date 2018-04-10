from keras.callbacks import Callback


class BatchLoss(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        return
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return