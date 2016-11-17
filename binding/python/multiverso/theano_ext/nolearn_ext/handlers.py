import multiverso
from ..lasagne_ext.param_manager import LasagneParamManager


class MVHandler:
    def __init__(self, freq=1):
        self.freq = freq
        self.cur_n = 0

    def on_training_started(self, nn, train_history=None):
        multiverso.barrier()
        self.kpm = LasagneParamManager(nn.layers_[-1])

    def on_batch_finished(self, nn, train_history=None):
        self.cur_n = (self.cur_n + 1) % self.freq
        if self.cur_n % self.freq == 0:
            self.kpm.sync_all_param()
