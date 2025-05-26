import os
import numpy as np
import time
import torch
import cv2

from utils import *

import vctoolkit as vc
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from loss import myLoss, myLoss25d, myLoss3d, l1_loss

from models.mobrecon_ds import *

from models.loss import pa_mpjpe_loss, p_mpjpe

import matplotlib.pyplot as plt
from utils import *

from utils_inter.mano import MANO


class Runner(object):
    def __init__(self, cfg, args, model, train_loader, eval_loader, test_loader, optimizer,
                                device, start_epoch=0, exp_name=None):
        super(Runner, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model = model
        self.device = device

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.optimizer = optimizer

        self.start_epoch = start_epoch
        self.max_epochs = cfg.TRAIN.EPOCHS


        if exp_name is None:
            self.folder = Path(os.path.join(self.cfg.SAVE.FOLDER_PATH,
                                        datetime.now().strftime('%m-%d_%H-%M')))
        else:
            self.folder = Path(os.path.join(self.cfg.SAVE.FOLDER_PATH,
                                        exp_name))
            self.ckpt_folder = Path(os.path.join(self.cfg.SAVE.FOLDER_PATH,
                                        exp_name, 'ckpt'))                       
        self.folder.mkdir(exist_ok=True)
        self.ckpt_folder.mkdir(exist_ok=True)

        self.folder_path = str(self.ckpt_folder)

        self.res_folder = Path(os.path.join(self.folder_path, 'result'))
        self.res_folder.mkdir(exist_ok=True)
        self.res_folder_path = str(self.res_folder)

        if exp_name is None:
            self.model_name = cfg.MODEL.NAME
        else:
            self.model_name = cfg.MODEL.NAME + f'_{exp_name}'

        # bring ckpt
        if start_epoch != 0:
            ckpt_root = os.path.join(self.folder_path, f'{self.model_name}_{start_epoch}.pth')
            ckpt = torch.load(ckpt_root)
            self.model.load_state_dict(ckpt, strict=False)
        else:
            ckpt_root = cfg.MODEL.PRETRAIN
            ckpt = torch.load(ckpt_root)
            print("Loading ", ckpt_root)
            self.model.load_state_dict(ckpt['model_state_dict'], strict=True)
        ml = MANO()
        self.j_reg = torch.from_numpy(ml.joint_regressor).unsqueeze(0).to(device) # [1, 21, 778]
        # if cfg.MODEL.PRETRAIN:
        #     self.model = load_model(self.model, cfg.MODEL.PRETRAIN)
           
        # if cfg.MODEL.BACKBONE_PRETRAIN:
        #     print(f'Loading Backbone from {cfg.MODEL.BACKBONE_PRETRAIN}')
        #     pretrain_path = cfg.MODEL.BACKBONE_PRETRAIN
        #     pretrain_dict = torch.load(pretrain_path)
        #     self.model.backbone.load_state_dict(pretrain_dict)

    def model_info(self, test_device='cpu'):
        from thop import profile, clever_format

        test_model = self.model.to(test_device)
        input = torch.randn((1, 3, self.cfg.IMG_SIZE, self.cfg.IMG_SIZE)).float().to(test_device)
        test_model.eval()
        output = test_model(input)
        macs, params = profile(self.model, inputs=(input,))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs, params)
        total = sum([param.nelement() for param in test_model.parameters()])
        print("Number of parameter: %.2fM" % (total/1e6))
        print("Checking Inference time: ")
        inf_time = 0
        with torch.no_grad():
            for step, data in tqdm(enumerate(range(0, 100))):
                st = time.time()
                out = test_model(input)
                et = time.time()
                inf_time += (et - st)
                

        print('Inference time: {:.3f}ms'.format(inf_time * 1000 / 100))
        
        self.model = self.model.to(self.device)

    def run(self):
        best_val = 1000
        self.loss = myLoss
        if self.train_loader:
            for epoch in range(self.start_epoch, self.max_epochs + 1):
                t = time.time()
                train_loss = self.train()
                print('train_loss: {}'.format(train_loss))
                if self.eval_loader is not None:
                    val_loss, val_mpjpe, val_pa_mpjpe = self.eval()
                    print(f'l1: {val_loss}, mpjpe: {val_mpjpe}, pa-mpjpe: {val_pa_mpjpe}')
                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save(self.model.state_dict(), os.path.join(self.folder_path, '{}_{}.pth'.format(self.model_name, epoch)))      
                if self.test_loader is not None:
                    self.pred()  
                elif True:
                # elif epoch % 2 == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.folder_path, '{}_{}.pth'.format(self.model_name, epoch)))       

                print('Epoch: {}'.format(epoch))
            torch.save(self.model.state_dict(), os.path.join(self.folder_path, '{}.pth'.format(self.model_name)))   

        if self.eval_loader:
            val_loss, val_mpjpe, val_pa_mpjpe = self.eval()
            print(f'l1: {val_loss}, mpjpe: {val_mpjpe}, pa-mpjpe: {val_pa_mpjpe}')
            torch.save(self.model.state_dict(), os.path.join(self.folder_path, '{}.pth'.format(self.model_name)))   
        
        if self.test_loader:
            self.pred()


    def run25d(self):
        best_val = 1000
        self.loss = myLoss25d
        if self.train_loader:
            for epoch in range(self.start_epoch, self.max_epochs + 1):
                t = time.time()
                train_loss = self.train_25d()
                print('train_loss: {}'.format(train_loss))
                if self.eval_loader is not None:
                    val_loss = self.eval25d()
                    print(f'l1: {val_loss}')
                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save(self.model.state_dict(), os.path.join(self.folder_path, '{}_{}.pth'.format(self.model_name, epoch))) 

                if self.test_loader:
                    self.pred()
                elif epoch % 2 == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.folder_path, '{}_{}.pth'.format(self.model_name, epoch)))       

                print('Epoch: {}'.format(epoch))
            torch.save(self.model.state_dict(), os.path.join(self.folder_path, '{}.pth'.format(self.model_name)))   

        if self.eval_loader:
            val_loss = self.eval25d()
            print(f'l1: {val_loss}')
            torch.save(self.model.state_dict(), os.path.join(self.folder_path, '{}.pth'.format(self.model_name)))   

    # for mobrecon origin model 240920
    def run3d(self):
        
        best_val = 1000
        self.loss = myLoss3d
        if self.train_loader:
            for epoch in range(self.start_epoch, self.max_epochs + 1):
                t = time.time()
                train_loss = self.train3d()
                print('train_loss: {}'.format(train_loss))
                if self.eval_loader is not None:
                    val_loss, val_mpjpe, val_pa_mpjpe = self.eval()
                    print(f'l1: {val_loss}, mpjpe: {val_mpjpe}, pa-mpjpe: {val_pa_mpjpe}')
                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save(self.model.state_dict(), os.path.join(self.folder_path, '{}_{}.pth'.format(self.model_name, epoch)))      
                if self.test_loader is not None:
                    self.pred()  
                elif (epoch+1) % 2 == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.folder_path, '{}_{}.pth'.format(self.model_name, epoch)))       

                print('Epoch: {}'.format(epoch))
            torch.save(self.model.state_dict(), os.path.join(self.folder_path, '{}.pth'.format(self.model_name)))   

        if self.eval_loader:
            val_loss, val_mpjpe, val_pa_mpjpe = self.eval()
            print(f'l1: {val_loss}, mpjpe: {val_mpjpe}, pa-mpjpe: {val_pa_mpjpe}')
            torch.save(self.model.state_dict(), os.path.join(self.folder_path, '{}.pth'.format(self.model_name)))   
        
        if self.test_loader:
            self.pred()

    def train3d(self, ):
        print("[TRAIN]")
        self.model.train()
        total_l1 = 0
        total_mpjpe = 0
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), smoothing=0.9)
        for step, data in pbar:
            for keys in data.keys():
                data[keys] = data[keys].to(self.device)
                
            input_img = data['image']
            gt_keypoints = data['keypoints3D']
            keypoints_2d = data['keypoints2D'] # noramlized [0, 1]

            self.optimizer.zero_grad()
            
            # print("model device: ", next(self.model.parameters()).device)
            # print(gt_keypoints.device)


            out = self.model(input_img)

            # loss cal
            loss_dict = dict()

            # Since mobrecon estimates verts -> change to joint
            bs = out['verts'].shape[0]

            pred_joints = torch.bmm(self.j_reg.repeat(bs, 1, 1), out['verts'])
            joint3d_loss = l1_loss(pred_joints, gt_keypoints)
            loss_dict['joint3d_loss'] = joint3d_loss.item()

            joint2d_loss = l1_loss(out['joint_img'], keypoints_2d)
            loss_dict['joint2d_loss'] = joint2d_loss.item()
            loss_dict['total'] = joint3d_loss + joint2d_loss

            # losses = self.loss(out, data)
            loss = loss_dict['total']
            # import pdb; pdb.set_trace()
            mpjpe = torch.mean(torch.linalg.norm((pred_joints - gt_keypoints), ord=2, dim=-1)) * 1000
            total_l1 += loss.item()
            total_mpjpe += mpjpe
            
            loss.backward()
            self.optimizer.step()
            pbar.set_description("t:{:.4f}, mpjpe:{:.4f}".format(loss_dict['total'].item(), mpjpe))
            
        return total_l1/ len(self.train_loader), total_mpjpe / len(self.train_loader)

    def train(self, ):
        print("[TRAIN]")
        self.model.train()
        total_l1 = 0
        total_mpjpe = 0
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), smoothing=0.9)
        for step, data in pbar:
            for keys in data.keys():
                data[keys] = data[keys].to(self.device)
                
            input_img = data['cropped_image']
            gt_keypoints = data['keypoints3D']
            
            self.optimizer.zero_grad()
            
            out = self.model(input_img)
            losses = self.loss(out, data)
            loss = losses['total']
            # import pdb; pdb.set_trace()
            mpjpe = torch.mean(torch.linalg.norm((out['keypoints3D'] - gt_keypoints), ord=2, dim=-1)) * 1000
            total_l1 += loss.item()
            total_mpjpe += mpjpe
            
            loss.backward()
            self.optimizer.step()
            pbar.set_description("t:{:.4f}, mpjpe:{:.4f}".format(losses['total'].item(), mpjpe))
            
        return total_l1/ len(self.train_loader), total_mpjpe / len(self.train_loader)
    
    def train_25d(self,):
        print("[TRAIN]")
        self.model.train()
        total_l1 = 0

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), smoothing=0.9)
        for step, data in pbar:
            for keys in data.keys():
                data[keys] = data[keys].to(self.device)
                
            input_img = data['cropped_image']
                       
            self.optimizer.zero_grad()
            
            out = self.model(input_img, torch.zeros(input_img.shape[0], 21, 3).to(self.device))
            
            losses = self.loss(out, data)
            loss = losses['total']
            # import pdb; pdb.set_trace()
            total_l1 += loss.item()
            
            loss.backward()
            self.optimizer.step()
            
            pbar.set_description("t:{:.4f}, l1:{:.4f}".format(losses['total'].item(), losses['total'].item()))

            if step > 0 and step % 1000 == 0:
                out_keypoints = out['keypoints'].cpu().detach()
                # out_keypoints3D = out['keypoints3D'].cpu().detach()
                B, C, H, W =  data['cropped_image'].size()
                for i in range(B):
                    res_img = draw_joint2D(data['cropped_image'], out_keypoints, idx=i)
                    cv2.imwrite(os.path.join(self.res_folder_path, '{}_{}_2d.jpg'.format(step, i)), cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

        return total_l1/ len(self.train_loader)


    def eval(self, ):
        print("[EVAL]")
        self.model.eval()
        mpjpe_total = 0
        pa_mpjpe_total = 0
        l1_total = 0
        
        total = len(self.eval_loader)
        pbar = tqdm(enumerate(self.eval_loader), total=len(self.eval_loader), smoothing=0.9)
        with torch.no_grad():
            for step, data in pbar:
                for keys in data.keys():
                    data[keys] = data[keys].to(self.device)
                
                input_img = data['cropped_image']
                gt_keypoints = data['keypoints3D']

                out = self.model(input_img,torch.zeros(input_img.shape[0], 21, 3).to(self.device))

                losses = self.loss(out, data)
                l1_total += losses['total'].item()

               #  import pdb; pdb.set_trace()
                mpjpe = torch.mean(torch.linalg.norm((out['keypoints3D'] - gt_keypoints), ord=2, dim=-1)) * 1000
                mpjpe_total += mpjpe
                
                pa_mpjpe = p_mpjpe(out['keypoints3D'], gt_keypoints) 
                pa_mpjpe_total += pa_mpjpe
                
                pbar.set_description("l1:{:.4f}, mpjpe:{:.4f}, pa mpjpe:{:.4f}".format(losses['total'].item(), mpjpe, pa_mpjpe))

        return l1_total/total, mpjpe_total / total, pa_mpjpe_total / total

    def eval25d(self, ):
        print("[EVAL]")
        self.model.eval()
        l1_total = 0
        
        total = len(self.eval_loader)
        pbar = tqdm(enumerate(self.eval_loader), total=len(self.eval_loader), smoothing=0.9)
        with torch.no_grad():
            for step, data in pbar:
                for keys in data.keys():
                    data[keys] = data[keys].to(self.device)
                
                input_img = data['cropped_image']
                gt_keypoints = data['keypoints']

                out = self.model(input_img, torch.zeros(input_img.shape[0], 21, 3).to(self.device))

                losses = self.loss(out, data)
                l1_total += losses['total'].item()
                
                if self.cfg.VAL.SAVE_PRED:
                    out_keypoints = out['keypoints']
                    res_img = draw_joint2D(data['cropped_image'], out_keypoints, idx=0)
                    cv2.imwrite(os.path.join(self.res_folder_path, '{}_2d.jpg'.format(step)), cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                    res_img = draw_joint2D(data['cropped_image'], gt_keypoints, idx=0)
                    cv2.imwrite(os.path.join(self.res_folder_path, '{}_gt.jpg'.format(step)), cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

                pbar.set_description("l1:{:.4f}".format(losses['total'].item()))

        return l1_total/total



    def pred(self, ):
        print("[TEST]")
        self.model.eval()

        inf_time = 0
        total_step = 0
        
        with torch.no_grad():
            for step, data in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                input_img = data['cropped_image'].to(self.device)
                st = time.time()
                out = self.model(input_img,torch.zeros(input_img.shape[0], 21, 3).to(self.device))
                et = time.time()
                inf_time += (et - st)
                total_step += 1


                # fig = plt.figure(figsize=(10, 10))
                # ax = plt.axes(projection = '3d')

                # res_img = data['cropped_image']
                # cv2.imwrite(os.path.join(self.res_folder_path, '{}_image.jpg'.format(step)), cv2.cvtColor(tensor2img(res_img), cv2.COLOR_RGB2BGR))
                # fig = plt.figure()
                # ax = plt.axes(projection = '3d')
                
                # gt_kpts3d = data['keypoints'][0].cpu().detach().numpy()
                # kpts3d = out['keypoints'][0].cpu().detach().numpy()
                
                # for finger, params in COLORMAP.items():
                #     for line in range(len(params['ids'])-1):
                #         i1, i2 = params['ids'][line], params['ids'][line+1]
                        
                #         x = np.array([gt_kpts3d[i1, 0], gt_kpts3d[i2, 0]])
                #         y = np.array([gt_kpts3d[i1, 1], gt_kpts3d[i2, 1]])
                #         z = np.array([gt_kpts3d[i1, 2], gt_kpts3d[i2, 2]])

                        
                #         ax.plot(x, z, -y, c='r', linewidth=1)
                # for finger, params in COLORMAP.items():
                #     for line in range(len(params['ids'])-1):
                #         i1, i2 = params['ids'][line], params['ids'][line+1]
                        
                #         x = np.array([kpts3d[i1, 0], kpts3d[i2, 0]])
                #         y = np.array([kpts3d[i1, 1], kpts3d[i2, 1]])
                #         z = np.array([kpts3d[i1, 2], kpts3d[i2, 2]])

                        
                #         ax.plot(x, z, -y, c='b', linewidth=1)
                        

                # plt.savefig(os.path.join(self.res_folder_path, '{}_gt_3d.jpg'.format(step)))
                # plt.close()
                if 'keypoints' in data.keys() and 'keypoints' in out.keys():
                    gt = data['keypoints']

                    out_keypoints = out['keypoints'].cpu()
                    # out_keypoints3D = out['keypoints3D'].cpu()
    
                    res_img = draw_joint2D(data['cropped_image'], out_keypoints, idx=0)
                    cv2.imwrite(os.path.join(self.res_folder_path, '{}_2d.jpg'.format(step)), cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                    res_img = draw_joint2D(data['cropped_image'], gt, idx=0)
                    cv2.imwrite(os.path.join(self.res_folder_path, '{}_gt.jpg'.format(step)), cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                # if 'keypoints3D' in data.keys() and 'keypoints3D' in out.keys():
                #     gt = data['keypoints3D']
                #     K = data['K']
                    
                #     out_keypoints = out['keypoints3D'].cpu()
    
                #     res_img = draw_joint(data['cropped_image'], out_keypoints, K, idx=0)
                #     cv2.imwrite(os.path.join(self.res_folder_path, '{}_3d.jpg'.format(step)), cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                #     res_img = draw_joint(data['cropped_image'], gt, K, idx=0)
                #     cv2.imwrite(os.path.join(self.res_folder_path, '{}_gt_3d.jpg'.format(step)), cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                if step > 9:
                    break
        print('Inference time: {:.5f}'.format(inf_time / 101))