#-*-coding:utf-8-*-

import torch


import time
import os

from tqdm import tqdm
from train_config import config as cfg



from lib.helper.logger import logger

from lib.core.model.loss.simpleface_loss import calculate_loss
from lib.core.base_trainer.metric import *

class Train(object):

  def __init__(self, model,train_ds,val_ds):
    self.epochs = cfg.TRAIN.epoch
    self.batch_size = cfg.TRAIN.batch_size
    self.l2_regularization=cfg.TRAIN.weight_decay_factor
    self.init_lr=cfg.TRAIN.init_lr


    self.save_dir=cfg.MODEL.model_path

    self.device = torch.device("cuda")

    self.model = model.to(self.device)

    self.load_weight()

    self.model = nn.DataParallel(self.model)
    if 'Adamw' in cfg.TRAIN.opt:

      self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                         lr=self.init_lr, eps=1.e-5,
                                         weight_decay=self.l2_regularization)
    else:
      self.optimizer = torch.optim.SGD(self.model.parameters(),
                                       lr=self.init_lr,
                                       momentum=0.9,
                                       weight_decay=self.l2_regularization)



    ###control vars
    self.iter_num=0


    self.train_ds=train_ds

    self.val_ds = val_ds

    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self.optimizer, self.epochs,eta_min=1.e-7)




  def loss_function(self,predict,label):

    loss=calculate_loss(predict,label)
    return loss

  def custom_loop(self):


    def distributed_train_epoch(epoch_num):

      summary_loss = AverageMeter()

      total_loss = 0.0
      num_train_batches = 0.0

      self.model.train()
      for images, target in self.train_ds:

        start=time.time()



        data  = images.to(self.device).float()
        target=target.to(self.device)
        batch_size = data.shape[0]

        output = self.model(data)


        current_loss = self.loss_function(output, target)
        self.optimizer.zero_grad()
        current_loss.backward()
        self.optimizer.step()

        total_loss += current_loss
        num_train_batches += 1
        self.iter_num+=1
        time_cost_per_batch=time.time()-start

        images_per_sec=cfg.TRAIN.batch_size/time_cost_per_batch

        summary_loss.update(current_loss.detach().item(), batch_size)

        if self.iter_num%cfg.TRAIN.log_interval==0:
          logger.info('epoch_num: %d, '
                      'iter_num: %d, '
                      'loss_value: %.6f,  '
                      'speed: %d images/sec ' % (epoch_num,
                                                 self.iter_num,
                                                 summary_loss.avg,
                                                 images_per_sec))

      return summary_loss

    def distributed_test_epoch(epoch_num):

      summary_loss = AverageMeter()

      total_loss=0.
      num_test_batches = 0.0
      self.model.eval()
      with torch.no_grad():
        for images, target in tqdm(self.val_ds):
          data = images.to(self.device).float()
          target = target.to(self.device)
          batch_size = data.shape[0]
          output = self.model(data)
          current_loss = self.loss_function(output, target)
          summary_loss.update(current_loss.detach().item(), batch_size)
      return summary_loss


    for epoch in range(self.epochs):

      for param_group in self.optimizer.param_groups:
        lr = param_group['lr']
      logger.info('learning rate: [%f]' % (lr))
      start=time.time()

      train_summary_loss= distributed_train_epoch(epoch)

      val_summary_loss = distributed_test_epoch(epoch)
      time_consume_per_epoch=time.time()-start
      training_massage = 'Epoch: %d, ' \
                         'Train Loss: %.6f, ' \
                         'Test Loss: %.6f '\
                         'Time consume: %.2f'%(epoch,
                                               train_summary_loss.avg,
                                               val_summary_loss.avg,
                                               time_consume_per_epoch)

      logger.info(training_massage)
      self.scheduler.step()

      #### save the model every end of epoch
      current_model_saved_name='%s/epoch_%d_val_loss%.6f.pth'%(self.save_dir,epoch,val_summary_loss.avg)

      logger.info('A model saved to %s' % current_model_saved_name)

      if not os.access(self.save_dir,os.F_OK):
        os.mkdir(self.save_dir)


      torch.save(self.model.module.state_dict(), current_model_saved_name)

      torch.cuda.empty_cache()

    return


  def load_weight(self):
    if cfg.MODEL.pretrained_model is not None:
      state_dict = torch.load(cfg.MODEL.pretrained_model, map_location=self.device)
      self.model.load_state_dict(state_dict, strict=False)