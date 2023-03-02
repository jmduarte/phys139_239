"""
Created on 7 Apr 2017

@author: jkiesele
"""
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class all_callbacks(object):
    def __init__(
        self, stop_patience=10, lr_factor=0.5, lr_patience=1, lr_epsilon=0.001, lr_cooldown=4, lr_minimum=1e-5, outputDir=""
    ):

        self.stopping = EarlyStopping(monitor="val_loss", patience=stop_patience, verbose=1, mode="min")

        self.reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_factor,
            patience=lr_patience,
            mode="min",
            verbose=1,
            min_delta=lr_epsilon,
            cooldown=lr_cooldown,
            min_lr=lr_minimum,
        )

        self.modelbestcheck = ModelCheckpoint(
            f"{outputDir}/model_best.h5", monitor="val_loss", verbose=1, save_best_only=True
        )

        self.modelcheck = ModelCheckpoint(f"{outputDir}/model_last.h5", verbose=1)

        self.callbacks = [self.modelbestcheck, self.modelcheck, self.reduce_lr, self.stopping]
