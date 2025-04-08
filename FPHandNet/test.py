import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import loss.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import os
from tqdm import tqdm
import torch.nn.functional as F
import shutil

def calc_eer_test(distances, label, threshold_list = None):
    '''
    计算等误率
    :param distances:  余弦距离矩阵，[batch_size]
    :param label:  标签，[batch_size]；1，表示同类；0，表示异类
    :return:
    '''
    # 将tensor转化为numpy
    distances_np = [item.numpy() for item in distances]
    distances_np = np.array(distances_np)
    label_np = np.array(label)

    batch_size = label_np.shape[0]
    minV = 100
    minV_2f = 100
    bestThresh = 0

    max_dist = np.max(distances_np)
    min_dist = np.min(distances_np)
    if not threshold_list:
        threshold_list = np.linspace(min_dist, max_dist, num=100)
    intra_cnt_final = 0
    inter_cnt_final = 0
    intra_len_final = 0
    inter_len_final = 0

    for threshold in (threshold_list):
        intra_cnt = 0
        intra_len = 0
        inter_cnt = 0
        inter_len = 0
        for i in (range(batch_size)):
            # intra
            if label_np[i] == 1:
                intra_len += 1
                if distances_np[i] < threshold:
                    intra_cnt += 1
            elif label_np[i] == 0:
                inter_len += 1
                if distances_np[i] > threshold:
                    inter_cnt += 1

        fr = intra_cnt / intra_len
        fa = inter_cnt / inter_len

        if abs(fr - fa) < minV:
            minV = abs(fr - fa)
            eer = (fr + fa) / 2
            bestThresh = threshold

            intra_cnt_final = intra_cnt
            inter_cnt_final = inter_cnt
            intra_len_final = intra_len
            inter_len_final = inter_len
        # far等于0.01
            if abs(0.01 - fa) < minV_2f:
                minV_2f = abs(0.01 - fa)
                tar_2f = 1 - fr
    # print('eer : {}, bestThresh : {},'.format(eer,bestThresh))
    # print("intra_cnt is : {} , inter_cnt is {} , intra_len is {} , inter_len is {}".format(intra_cnt_final,inter_cnt_final,intra_len_final,inter_len_final))

    return intra_cnt_final,inter_cnt_final,intra_len_final,inter_len_final,eer, bestThresh, minV

def valid_epoch(model,valid_data_loader,test_data_loader,debug = False):
    """
    Validate after training an epoch

    :param epoch: Integer, current training epoch.
    :return: A log that contains information about validation
    """
    # 遍历测试集,获取所有特征
    features = {}
    with torch.no_grad():
        for batch_idx, (data,name) in enumerate(tqdm(test_data_loader)):
            data[0] = data[0].cuda()
            data[1] = data[1].cuda()
            output_thumb,output_fore,output_middle,output_ring,output_little,output_palm,output_cat,*_ = model(data)
            output = output_cat.cpu()
            for i in range(output.shape[0]):
                features[name[i]] = output[i]
    distances = []
    label = []
    distances_save = []
    for item in test_data_loader.dataset.query:
        dis = F.cosine_similarity(features[item[0]].unsqueeze(0), features[item[1]].unsqueeze(0))
        distances.append(dis)
        distances_save.append(dis.item())
        label.append(item[2])
    intra_cnt_final,inter_cnt_final,intra_len_final,inter_len_final,test_eer, bestThresh, minV = calc_eer_test(distances, label)
    print("test_eer",test_eer)
    
    return test_eer

def main(config):
    # setup data_loader instances
    train_data_loader = config.init_obj('train_FP_data_loader', module_data)
    valid_data_loader = config.init_obj('valid_FP_data_loader', module_data)
    test_data_loader = config.init_obj('test_FP_data_loader', module_data)
    
    # build model architecture
    model = config.init_obj('arch', module_arch)
    print("module_arch",module_arch)
    
    # 类别一致可以加载进来的情况在这里 -l
    if not config.resume == None:
        print('Loading checkpoint: {} ...'.format(config.resume))
        checkpoint = torch.load(config.resume)

        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        
        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
        model = model.to(device)
        
        print("str(config.resume).split('/')[-2]",str(config.resume).split('/')[-2])
        parts = (str(config.resume).split('/')[-2]).split("_",2)


        debug = str(config.resume).split('/')[-2]#parts[-1] if len(parts)==3 else str(config.resume).split('/')[-2]
    
    model.eval()
    test_eer = valid_epoch(model,valid_data_loader,test_data_loader,debug)
    print(test_eer)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config/Tongji_inscribe_palmprint_config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-l', '--log', default=None, type=str,
                    help='log name')
    args.add_argument('-D', '--dir', default="0225_103653_Tongji_palmprint_arcface", type=str,
                    help='log name')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
