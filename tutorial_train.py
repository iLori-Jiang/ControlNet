from share import *

import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from pytorch_lightning.loggers import WandbLogger
import wandb


# Configs
model_name = 'control_toy_sd15_step=12000'
resume_path = './models/' + model_name + '.ckpt'
batch_size = 4
accumulate_grad_batches = 4
logger_freq = 300
learning_rate = 1e-5
training_samples = 4000*4*2
train_steps = training_samples / batch_size
sd_locked = True
only_mid_control = False


# 初始化WandB
wandb_logger = WandbLogger(project='Finetune ControlNet on toy dataset')


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

root_dir = os.path.dirname(os.path.abspath(__file__))
image_logger = ImageLogger(save_dir=root_dir, batch_frequency=logger_freq)


# 自定义回调以确保在训练结束时保存模型
class SaveCheckpointOnFinish(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        trainer.save_checkpoint("lightning_logs/" + model_name + "-last.ckpt")


# 设置检查点回调，保存最佳模型和每个epoch末尾的模型
checkpoint_callback = ModelCheckpoint(
    dirpath='lightning_logs/',
    filename= model_name + '-{epoch}-{step}',
    save_top_k=1,
    every_n_train_steps = logger_freq
)

# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
trainer = pl.Trainer(
            max_steps=train_steps,
            accumulate_grad_batches=accumulate_grad_batches,
            devices=[0], 
            accelerator='gpu', 
            precision=32, 
            callbacks=[image_logger, checkpoint_callback, SaveCheckpointOnFinish()],
            logger = wandb_logger
        )


# Train!
trainer.fit(model, dataloader)
