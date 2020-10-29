import torch
import torch.nn as nn
import torch.nn.functional as F
from . import counters

class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name, dcl_conf):
        super(CrowdCounter, self).__init__()        

        net =  getattr(getattr(counters, model_name), model_name)

        self.CCN = net()
        if len(gpus)>1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN=self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()

        self.dcl_conf = dcl_conf
        self.age = dcl_conf.stAge
        if dcl_conf.radius > 0:
            radius = dcl_conf.radius
            self.radGen = nn.AvgPool2d(radius * 2 + 1, stride=1, padding=radius)

    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, img, gt_map, train=False):
        train = self.dcl_conf.work and train and self.age <= self.dcl_conf.end_epoch
        density_map = self.CCN(img)
        gt_map = gt_map.unsqueeze(1)
        if train:
            att_maps = self.denGauss(gt_map)
            density_map = density_map * att_maps
            gt_map = gt_map * att_maps
        self.loss_mse = self.build_loss(density_map, gt_map)
        return density_map
    
    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)  
        return loss_mse

    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map
    
    def denGauss(self, x):
        dcl_conf = self.dcl_conf
        gauss = lambda x, mu, sigma: torch.exp(-((x - mu) / sigma) ** 2)
        if dcl_conf.radius > 0:
            x = self.radGen(x)
        weight = gauss(x, dcl_conf.mu, self.age * dcl_conf.alpha + dcl_conf.beta) * dcl_conf.rate
        return weight

    def updateAge(self, age=1):
        self.age += age

