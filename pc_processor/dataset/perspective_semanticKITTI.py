import numpy as np
import torch
from torch.utils.data import Dataset
from pc_processor.dataset.preprocess import augmentor
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import yaml
import cv2

class semanticKITTI_refine_PerspectiveViewLoader(Dataset):
    def __init__(self, dataset, config, data_len=-1, is_train=True, pcd_aug=False, img_aug=False, use_padding=False,
                 return_uproj=False, max_points=150000):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.pcd_aug = pcd_aug
        self.img_aug = img_aug
        self.data_len = data_len
        self.use_padding = use_padding

        if not self.is_train:
            self.pcd_aug = False
            self.img_aug = False
        augment_params = augmentor.AugmentParams()
        augment_config = self.config['augmentation']

        if self.pcd_aug:
            augment_params.setFlipProb(
                p_flipx=augment_config['p_flipx'], p_flipy=augment_config['p_flipy'])
            augment_params.setTranslationParams(
                p_transx=augment_config['p_transx'], trans_xmin=augment_config[
                    'trans_xmin'], trans_xmax=augment_config['trans_xmax'],
                p_transy=augment_config['p_transy'], trans_ymin=augment_config[
                    'trans_ymin'], trans_ymax=augment_config['trans_ymax'],
                p_transz=augment_config['p_transz'], trans_zmin=augment_config[
                    'trans_zmin'], trans_zmax=augment_config['trans_zmax'])
            augment_params.setRotationParams(
                p_rot_roll=augment_config['p_rot_roll'], rot_rollmin=augment_config[
                    'rot_rollmin'], rot_rollmax=augment_config['rot_rollmax'],
                p_rot_pitch=augment_config['p_rot_pitch'], rot_pitchmin=augment_config[
                    'rot_pitchmin'], rot_pitchmax=augment_config['rot_pitchmax'],
                p_rot_yaw=augment_config['p_rot_yaw'], rot_yawmin=augment_config[
                    'rot_yawmin'], rot_yawmax=augment_config['rot_yawmax'])
            self.augmentor = augmentor.Augmentor(augment_params)
        else:
            self.augmentor = None

        if self.img_aug:
            self.img_jitter = transforms.ColorJitter(
                *augment_config["img_jitter"])
        else:
            self.img_jitter = None

        projection_config = self.config['sensor']

        if self.use_padding:
            h_pad = projection_config["h_pad"]
            w_pad = projection_config["w_pad"]
            self.pad = transforms.Pad((w_pad, h_pad))
        else:
            h_pad = 0
            w_pad = 0

        self.return_uproj = return_uproj
        self.max_points = max_points

    def __getitem__(self, index):
        pointcloud, sem_label, _ = self.dataset.loadDataByIndex(index)

        if self.pcd_aug:
            pointcloud = self.augmentor.doAugmentation(pointcloud)

        image = self.dataset.loadImage(index)
        if self.img_aug:
            image = self.img_jitter(image)
        image = np.array(image)

        seq_id, _ = self.dataset.parsePathInfoByIndex(index)

        mapped_pointcloud, keep_mask = self.dataset.mapLidar2Camera(
            seq_id, pointcloud[:, :3], image.shape[1], image.shape[0])
        mapped_pointcloud = mapped_pointcloud.astype(np.int32)

        if not self.return_uproj:
            if self.is_train:
                after_h = self.config['sensor']['proj_ht'] - 2 * self.config['sensor']['h_pad']
                after_w = self.config['sensor']['proj_wt'] - 2 * self.config['sensor']['w_pad']
                image = image[:after_h, :after_w, :]
                keep_idx_pts = (mapped_pointcloud[:, 0] > 0) * (mapped_pointcloud[:, 0] < after_h) * (
                        mapped_pointcloud[:, 1] > 0) * (mapped_pointcloud[:, 1] < after_w)
                keep_mask[keep_mask] = keep_idx_pts
                mapped_pointcloud = mapped_pointcloud[keep_mask]

            else:
                after_h = self.config['sensor']['proj_h'] - 2 * self.config['sensor']['h_pad']
                after_w = self.config['sensor']['proj_w'] - 2 * self.config['sensor']['w_pad']
                image = image[:after_h, :after_w, :]
                keep_idx_pts = (mapped_pointcloud[:, 0] > 0) * (
                            mapped_pointcloud[:, 0] < after_h) * (
                                       mapped_pointcloud[:, 1] > 0) * (
                                           mapped_pointcloud[:, 1] < after_w)
                keep_mask[keep_mask] = keep_idx_pts
                mapped_pointcloud = mapped_pointcloud[keep_mask]

        unproj_sem_label = torch.full([self.max_points], -1.0, dtype=torch.int32)
        unproj_sem_label[:mapped_pointcloud.shape[0]] = torch.from_numpy(self.dataset.labelMapping(sem_label[keep_mask]))

        y_data = mapped_pointcloud[:, 1]
        x_data = mapped_pointcloud[:, 0]

        image = image.astype(np.float32) / 255.0

        depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)
        keep_poincloud = pointcloud[keep_mask]

        unproj_keep_poincloud = torch.full((self.max_points, 4), -1.0, dtype=torch.float)
        unproj_keep_poincloud[:keep_poincloud.shape[0]] = torch.from_numpy(keep_poincloud)
        n_points = keep_poincloud.shape[0]

        proj_xyzi = np.zeros(
            (image.shape[0], image.shape[1], keep_poincloud.shape[1]), dtype=np.float32)
        proj_xyzi[x_data, y_data] = keep_poincloud
        proj_depth = np.zeros(
            (image.shape[0], image.shape[1]), dtype=np.float32)
        proj_depth[x_data, y_data] = depth[keep_mask]

        proj_label = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
        try:
            proj_label[x_data,  y_data] = self.dataset.labelMapping(sem_label[keep_mask])
        except Exception as msg:
            print(msg)
            print(keep_mask.shape)
            print(sem_label.shape)

        proj_mask = np.zeros(
            (image.shape[0], image.shape[1]), dtype=np.int32)
        proj_mask[x_data, y_data] = 1

        image_tensor = torch.from_numpy(image)
        proj_depth_tensor = torch.from_numpy(proj_depth)
        proj_xyzi_tensor = torch.from_numpy(proj_xyzi)
        proj_label_tensor = torch.from_numpy(proj_label)
        proj_mask_tensor = torch.from_numpy(proj_mask)

        proj_tensor = torch.cat(
            (proj_depth_tensor.unsqueeze(0), # 1,h,w
             proj_xyzi_tensor.permute(2, 0, 1), # c,h,w
             image_tensor.permute(2, 0, 1), # c,h,w
             proj_mask_tensor.float().unsqueeze(0), # 1,h,w
             proj_label_tensor.float().unsqueeze(0)), dim=0) # 1,h,w

        if self.return_uproj:
            unproj_mapped_pointcloud = torch.full((self.max_points, 2), -1.0, dtype=torch.long)
            unproj_mapped_pointcloud[:mapped_pointcloud.shape[0]] = torch.from_numpy(mapped_pointcloud)

            return proj_tensor[:8], proj_tensor[8], proj_tensor[9], unproj_mapped_pointcloud, unproj_keep_poincloud, unproj_sem_label,n_points,torch.from_numpy(depth)

        else:
            if self.use_padding:
                proj_tensor = self.pad(proj_tensor)

            mapped_pointcloud[:,0] = mapped_pointcloud[:, 0] + self.config['sensor']['h_pad']
            mapped_pointcloud[:,1] = mapped_pointcloud[:, 1] + self.config['sensor']['w_pad']

            unproj_mapped_pointcloud = torch.full((self.max_points, 2), -1.0,dtype=torch.long)
            unproj_mapped_pointcloud[:mapped_pointcloud.shape[0]] = torch.from_numpy(mapped_pointcloud)

            return proj_tensor[:8], proj_tensor[8], proj_tensor[9], unproj_mapped_pointcloud, unproj_keep_poincloud, unproj_sem_label, n_points# proj_tensor[:8]的形状为[8(d,x,y,z,i,R,G,B),proj_ht,proj_wt], proj_tensor[8]为[proj_ht,proj_wt]是点云投影到图像像素位置的mask，proj_tensor[9]为[proj_ht,proj_wt]是点云投影图像对应的语义label

    def __len__(self):
        if self.data_len > 0 and self.data_len < len(self.dataset):
            return self.data_len
        else:
            return len(self.dataset)