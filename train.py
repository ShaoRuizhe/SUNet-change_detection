import numpy as np
import os

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader,random_split
from model import SUNnet

import cv2
import logging
import datetime
import sys



DATA_SET='HTCD'
data_dir = '%path_to_dataset%/tiles'
MODEL='SUNet'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPOCHES=300


# 01-07是chisinau-FC_EF的模型 # 加edge修改
# resume_model='logger/' + DATA_SET + '_' + MODEL + '_01-07'+ '/weights/model_para_52.pth'
resume_model=None


class HTCD(Dataset):
    # img1-sat img2-uav
    def __init__(self, dir_chin_data):
        # ls_pick_images:选中的大图号（int）的list,方便划分数据集之用
        self.dir = dir_chin_data
        self.images = os.listdir(os.path.join(self.dir,'uav'))
        self.sat_mean = np.array([66, 71, 74], np.uint8)
        self.uav_mean = np.array([73, 81, 79], np.uint8)

    def __getitem__(self, idx):
        # img1-sat img2-uav
        filename = self.images[idx]
        img1_file = os.path.join(self.dir, 'sat', filename)
        edge1_file = os.path.join(self.dir, 'edges_uav', filename + '.jpg')
        img2_file = os.path.join(self.dir, 'uav', filename)
        edge2_file=os.path.join(self.dir, 'edges_sat', filename + '.jpg')
        lbl_file = os.path.join(self.dir, 'label', filename)

        img1 = cv2.imread(img1_file).astype(np.int)
        img1 -= self.sat_mean
        if (img1 is None):
            print(idx)
            print(img1_file)
        img_size = img1.shape[:2]
        edge1=cv2.imread(edge1_file, cv2.IMREAD_UNCHANGED)
        edge1 = cv2.resize(edge1, img_size).astype(np.int)
        img1=np.concatenate((img1, edge1[..., np.newaxis]), axis=2)
        img1 = img1.transpose((2, 0, 1)).astype(np.float32) / 128

        img2 = cv2.imread(img2_file)
        img2 = cv2.resize(img2, (2048,2048)).astype(np.int)
        img2 -= self.uav_mean
        edge2=cv2.imread(edge2_file, cv2.IMREAD_UNCHANGED)
        edge2 = cv2.resize(edge2, (2048,2048)).astype(np.int)
        img2 = np.concatenate((img2, edge2[..., np.newaxis]), axis=2)
        img2 = img2.transpose((2, 0, 1)).astype(np.float32) / 128

        lbl = cv2.imread(lbl_file, cv2.IMREAD_UNCHANGED)
        lbl = cv2.resize(lbl, img_size)
        lbl = np.asarray(lbl)

        return img1, img2, lbl

    def __len__(self):
        return len(self.images)


class LossTotal(nn.Module):
    def __init__(self,weight_ba_loss,weight_ce_loss):
        super(LossTotal,self).__init__()
        self.bn=nn.BatchNorm2d(num_features=1)
        torch.nn.init.constant(self.bn.weight, 1)
        self.bn.to(device)
        self.ce_loss=nn.CrossEntropyLoss(weight=torch.FloatTensor([1,36]).to(device=device))
        self.weight_ba_loss = weight_ba_loss
        self.weight_ce_loss = weight_ce_loss

    def forward(self,y,lbl):
        ce_loss = self.ce_loss(y, lbl)
        diff = y[:, 1] - y[:, 0]  # 第1维大的为changed
        diff = torch.unsqueeze(diff, 1)
        diff = self.bn(diff)
        diff = torch.sigmoid(diff)
        lbl_float = lbl.float()
        iou_loss = 1-torch.sum(diff * lbl_float) / torch.sum(diff + lbl_float - diff * lbl_float)
        loss = iou_loss * self.weight_ba_loss+ce_loss*self.weight_ce_loss
        return loss

def main():
    strtime = datetime.datetime.now().strftime('%m-%d')
    log_dir = 'logger/' + DATA_SET + '_' + MODEL + '_' + strtime + '-1'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logging.basicConfig(filename=log_dir + '/logging.log', level=logging.INFO,

                        format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    from tensorboardX import SummaryWriter

    my_log_info='training SUNet with HTCD dataset\nlogdir:'+log_dir

    writer = SummaryWriter(log_dir + '/TensorBoard')
    writer.add_text(tag='my_log_info',text_string=my_log_info)
    weights_dir=log_dir+'/weights'
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    logger=logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.info(my_log_info)

    batch_size = 5
    lr = 0.005
    lr_step_size = 1000
    lr_decay = 1
    val_ratio = 0.2
    dataset = HTCD(data_dir)
    train_data, validation_data = random_split(dataset, [round((1 - val_ratio) * len(dataset)),
                                                         round(val_ratio * len(dataset))])
    logging.info('training set:%d patches' % len(train_data))
    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
    logging.info('validation set:%d patches' % len(validation_data))
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size,
                                       shuffle=False, num_workers=4, pin_memory=True)
    # 用于tensorboard画图的input tensor
    rand_tensor_t0 = torch.rand(1, 4, 256, 256).to(device, dtype=torch.float)
    rand_tensor_t1 = torch.rand(1, 4, 2048, 2048).to(device, dtype=torch.float)


    model = SUNnet().to(device, dtype=torch.float)
    if resume_model!=None:
        checkpoint=torch.load(resume_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info('resume success')
    writer.add_graph(model,(rand_tensor_t0,rand_tensor_t1))


    weight_ba_loss = 0.67  # iou_loss的权值
    weight_ce_loss = 0.33  # cross_entropy_loss的权值
    momentum=0.9
    weight_decay=0.0005
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=lr_step_size,gamma=lr_decay)
    loss_t=LossTotal(weight_ba_loss=weight_ba_loss,weight_ce_loss=weight_ce_loss)
    #每个epoch记录一次trainloss,计算一次validation loss,没100个batch记录一次100平均loss
    ave_loss_total=[]
    ave_loss_validation=[]
    ave_loss_100 = []

    logger.info('training ready.MetaData:\n lr:%f,lr_step_size:%d,lr_decay:%f,momentum:%f,weight_decay:%f\n'
                'weight_ba_loss:%f,weight_ce_loss:%f\nval_radio:%f'
                %(lr,lr_step_size,lr_decay,momentum,weight_decay,weight_ba_loss,weight_ce_loss,val_ratio))
    for epoch in range(EPOCHES):

        loss_100=[]
        loss_total=[]
        model.train()
        union_total=0
        intersection_total=0
        for i, data in enumerate(train_dataloader):
            x1, x2, lbl = data
            x1 = x1.to(device, dtype=torch.float)
            x2 = x2.to(device, dtype=torch.float)
            lbl = lbl.to(device, dtype=torch.long)
            y = model(x1,x2)
            optimizer.zero_grad()
            loss = loss_t(y, lbl)
            loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch)
            loss_100.append(loss.item())
            loss_total.append(loss.item())
            pre_label = y[:, 0] < y[:, 1]  # 第1维大的为 changed
            intersection = pre_label[lbl == 1].long().sum()
            union = pre_label.sum() + lbl.sum() - intersection
            intersection_total += intersection
            union_total += union
            if(i%100==0 and i>0):
                mean_loss=np.mean(loss_100)
                writer.add_scalar('loss_100',mean_loss,global_step=len(ave_loss_100))
                logging.info('average loss of batch '+str(i-99)+'-'+str(i)+':'+str(mean_loss))
                ave_loss_100.append(mean_loss)
                loss_100 = []
        mean_loss=np.mean(loss_total)
        writer.add_scalar('loss_total',mean_loss,global_step=epoch)
        iou=(intersection_total.float()/union_total.float()).cpu().numpy()
        writer.add_scalar('iou_train',iou,global_step=epoch)
        logging.info('average loss of epoch'+str(epoch)+':'+str(mean_loss))
        logging.info('average train iou of epoch ' + str(epoch) + ': ' + str(iou))
        ave_loss_total.append(mean_loss)
        # validation
        loss_total = []
        intersection_total = 0
        union_total = 0
        TP_total = 0
        TN_total = 0
        FP_total = 0
        FN_total = 0
        model.eval()
        for i, data in enumerate(validation_dataloader):
            x1, x2, lbl = data
            x1 = x1.to(device, dtype=torch.float)
            x2 = x2.to(device, dtype=torch.float)
            lbl = lbl.to(device, dtype=torch.long)
            y = model(x1, x2)
            loss = loss_t(y, lbl)
            loss_total.append(loss.item())
            pre_label = y[:, 0] < y[:, 1]  # 第1维大的为真
            TP = intersection = pre_label[lbl == 1].long().sum()
            union = pre_label.sum() + lbl.sum() - intersection
            intersection_total += intersection.item()
            union_total += union.item()
            FN = (1 - pre_label)[lbl == 1].long().sum()
            TN = (1 - pre_label)[lbl == 0].long().sum()
            FP = pre_label[lbl == 0].long().sum()
            TP_total += TP.item()
            TN_total += TN.item()
            FP_total += FP.item()
            FN_total += FN.item()

        mean_loss = np.mean(loss_total)
        writer.add_scalar('loss_validation', mean_loss, global_step=epoch)
        lbl_total = FP_total + TP_total + TN_total + FN_total
        precision = TP_total / (TP_total + FP_total+0.01)
        recall = TP_total / (TP_total + FN_total+0.01)
        F1 = 2 * precision * recall / (precision + recall+0.01)
        OA = (TP_total + TN_total) / (lbl_total)
        iou = float(intersection_total) / (union_total+0.01)
        metric_msg = "diff_lbl_sum:%d,precision:%.5f,recall:%.5f,F1 score:%.5f,OA:%.5f,iou:%.5f" % \
                     (lbl_total - FP_total - TP_total - TN_total - FN_total, precision, recall, F1, OA, iou)
        logging.info('validation metrics of epoch ' + str(epoch) + metric_msg)
        writer.add_scalar('iou_validation', iou, global_step=epoch)
        ave_loss_validation.append(mean_loss)
        logging.info('validation loss:' + str(mean_loss))

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, weights_dir + '/model_para_{}.pth'.format(epoch))


if __name__ == '__main__':
    main()

