from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os

import torch
import torch.utils.data
from opts import opts
from model.model import create_model, load_model, save_model
from model.data_parallel import DataParallel
from logger import Logger
from dataset.dataset_factory import get_dataset
from trainer import Trainer
from test import prefetch_test
import json

def get_optimizer(opt, model):
  if opt.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  elif opt.optim == 'sgd':
    print('Using SGD')
    optimizer = torch.optim.SGD(
      model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
  else:
    assert 0, opt.optim
  return optimizer

def main(opt):
  torch.manual_seed(opt.seed)  # 设置随机种子，opt.seed 默认为 317
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.eval # 是否使用 cudnn.benchmark
    # ，由于两个值都是false, 所以使用 cudnn.benchmark, cudnn.benchmark 会根据输入数据的大小和类型自动选择最适合的卷积算法
    # 对于小数据集，可以令该值为false，可以减少GPU内存的使用 
  Dataset = get_dataset(opt.dataset)  # 获取数据集，opt.dataset 默认为 'nuScenes', 返回的是 nuScenes 对象
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)  # 更新 opt 中的数据集信息，并设置头部信息
  print(opt)
  if not opt.not_set_cuda_env:  # 如果 opt.not_set_cuda_env 为 False
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str  # 设置 CUDA_VISIBLE_DEVICES 环境变量，opt.gpus_str = opt.gpus 的字符串形式
  opt.device = torch.device('cuda:0' if opt.gpus[0] >= 0 else 'cpu')  # 如果 opt.gpus[0] 大于等于 0，则设置设备为 cuda（即
    # GPU）；否则，设置设备为 cpu。
  logger = Logger(opt)

  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)  # 创建模型，opt.arch默认为 'dla_34', opt.heads 默认为 {
    # 'hm': 80, 'wh': 2, 'reg': 2}，opt.head_conv 默认为 -1(DLA使用256通道)
  optimizer = get_optimizer(opt, model)  # 获取优化器
  start_epoch = 0
  lr = opt.lr  # 学习率
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, opt, optimizer)  # 加载预训练模型

  trainer = Trainer(opt, model, optimizer)  # 根据参数、模型、优化器得到对应任务的训练器（其中包括损失统计、模型损失等
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)  # 设置设备
  if opt.val_intervals < opt.num_epochs or opt.eval:  # 验证间隔小于总训练轮数，则设置验证数据
    print('Setting up validation data...')
    val_loader = torch.utils.data.DataLoader(
      Dataset(opt, opt.val_split), batch_size=1, shuffle=False, 
              num_workers=1, pin_memory=True)  # 创建验证数据加载器

    if opt.eval:  # 命令行不包含，这个值为false
      _, preds = trainer.val(0, val_loader)  # 验证
      val_loader.dataset.run_eval(preds, opt.save_dir, n_plots=opt.eval_n_plots, 
                                  render_curves=opt.eval_render_curves)  # 运行评估
      return

    # 设置训练数据
  print('Setting up train data...')
  # 创建训练数据加载器, 如果数据集大小不能被 batch size 整除，则设置为 True 后可删除最后一个不完整的 batch。
    # 如果设为 False ，则最后一个 batch 将更小。(默认: False)
  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, opt.train_split), batch_size=opt.batch_size, 
        shuffle=opt.shuffle_train, num_workers=opt.num_workers, 
        pin_memory=True, drop_last=True
  )

  print('Starting training...')
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'  # 是否每 5 个 epoch 保存模型到磁盘，由于 train.sh 中没有添加 save_all 参数，所以为 False，最后
        # mark = 'last'，意为最后再保存模型到磁盘中

    # log learning rate
    for param_group in optimizer.param_groups:
      lr = param_group['lr']
      logger.scalar_summary('LR', lr, epoch)
      break
    
    # train one epoch
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    
    '''
            训练一轮模型，返回 ret 和 result（ret是一个字典：损失项--损失值。result也是一个字典：图像ID--检测结果）
            这里 log_dict_train = ret
            train()：/src/lib/trainer.py 第 405 行
    '''
    # log train results
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    
    # evaluate
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:  # 如果验证间隔大于 0 且 epoch 能被验证间隔整除，执行验证
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():  # 后面的部分禁用梯度计算
        log_dict_val, preds = trainer.val(epoch, val_loader)
        
        # evaluate val set using dataset-specific evaluator
        if opt.run_dataset_eval:
          out_dir = val_loader.dataset.run_eval(preds, opt.save_dir, 
                                                n_plots=opt.eval_n_plots, 
                                                render_curves=opt.eval_render_curves)
          
          # log dataset-specific evaluation metrics
          with open('{}/metrics_summary.json'.format(out_dir), 'r') as f:
            metrics = json.load(f)
          logger.scalar_summary('AP/overall', metrics['mean_ap']*100.0, epoch)
          for k,v in metrics['mean_dist_aps'].items():
            logger.scalar_summary('AP/{}'.format(k), v*100.0, epoch)
          for k,v in metrics['tp_errors'].items():
            logger.scalar_summary('Scores/{}'.format(k), v, epoch)
          logger.scalar_summary('Scores/NDS', metrics['nd_score'], epoch)
      
      # log eval results
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
    
    # save this checkpoint
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.save_point:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
    
    # update learning rate
    if epoch in opt.lr_step:
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

  logger.close()

if __name__ == '__main__':
  opt = opts().parse()  # opts() 创建了一个 opts 类的实例，parse() 方法解析命令行参数，并返回一个包含所有参数的对象 opt
  main(opt)  # 调用 main() 函数，传入参数 opt
