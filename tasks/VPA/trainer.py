import numpy as np
import os
import torch
import time
from option import Option
import torch.nn as nn
import datetime
import pc_processor
import math
import torch.nn.functional as F
import cv2
import yaml
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate
from torchsparse import SparseTensor
from pc_processor.PointRefine.spvcnn import SPVCNN

class Trainer(object):
    def __init__(self, settings: Option, model: nn.Module, recorder=None):
        self.settings = settings
        self.recorder = recorder
        self.remain_time = pc_processor.utils.RemainTime(
            self.settings.n_epochs)

        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = self._initDataloader()

        self.refine_module = SPVCNN(num_classes=self.settings.nclasses,
                                    cr=1.0,
                                    pres=0.05,
                                    vres=0.05)
        self.refine_module.cuda()
        self.model = model.cuda()

        # init criterion
        self.criterion = self._initCriterion()
    
        # init optimizer
        [self.optimizer, self.aux_optimizer,self.refine_optimizer] = self._initOptimizer()

        # get metrics for pcd
        self.metrics = pc_processor.metrics.IOUEval(
            n_classes=self.settings.nclasses, device=torch.device("cpu"),
            ignore=self.ignore_class, is_distributed=self.settings.distributed)
        self.metrics.reset()

        # get metrics for img
        self.metrics_img = pc_processor.metrics.IOUEval(
            n_classes=self.settings.nclasses, device=torch.device("cpu"),
            ignore=self.ignore_class, is_distributed=self.settings.distributed)
        self.metrics_img.reset()

        self.scheduler = pc_processor.utils.WarmupCosineLR(
            optimizer=self.optimizer,
            lr=self.settings.lr,
            warmup_steps=self.settings.warmup_epochs *
            len(self.train_loader),
            momentum=self.settings.momentum,
            max_steps=len(self.train_loader) * (self.settings.n_epochs-self.settings.warmup_epochs))

        self.refine_scheduler = pc_processor.utils.WarmupCosineLR(
            optimizer=self.refine_optimizer,
            lr=self.settings.lr,
            warmup_steps=self.settings.warmup_epochs *
                         len(self.train_loader),
            momentum=self.settings.momentum,
            max_steps=len(self.train_loader) * (self.settings.n_epochs - self.settings.warmup_epochs))

        self.aux_scheduler = pc_processor.utils.WarmupCosineLR(
            optimizer=self.aux_optimizer,
            lr=self.settings.lr,
            warmup_steps=self.settings.warmup_epochs *
            len(self.train_loader),
            momentum=self.settings.momentum,
            max_steps=len(self.train_loader) * (self.settings.n_epochs-self.settings.warmup_epochs))

    def _initOptimizer(self):
        # check params
        adam_params = [{"params": self.model.lidar_stream.parameters()}]

        adam_opt = torch.optim.AdamW(
            params=adam_params, lr=self.settings.lr)

        refine_optimizer = torch.optim.Adam(
            params=[{'params': self.refine_module.parameters()}], lr=self.settings.lr)

        sgd_params = [
            {"params": self.model.camera_stream_encoder.parameters()},
            {"params": self.model.camera_stream_decoder.parameters()}]

        sgd_opt = torch.optim.SGD(
            params=sgd_params, lr=self.settings.lr,
            nesterov=True,
            momentum=self.settings.momentum,
            weight_decay=self.settings.weight_decay)

        optimizer = [adam_opt, sgd_opt, refine_optimizer]
        return optimizer

    def _initDataloader(self):
        if self.settings.dataset == "SemanticKitti":
            data_config_path = "../../pc_processor/dataset/semantic_kitti/semantic-kitti.yaml"

            trainset = pc_processor.dataset.semantic_kitti.SemanticKitti(
                root=self.settings.data_root,
                sequences=[0,1,2,3,4,5,6,7,9,10],
                config_path=data_config_path
            )
            self.cls_weight = 1 / (trainset.cls_freq + 1e-3)
            self.ignore_class = []
            for cl, w in enumerate(self.cls_weight):
                if trainset.data_config["learning_ignore"][cl]:
                    self.cls_weight[cl] = 0
                if self.cls_weight[cl] < 1e-10:
                    self.ignore_class.append(cl)
            if self.recorder is not None:
                self.recorder.logger.info("weight: {}".format(self.cls_weight))
            self.mapped_cls_name = trainset.mapped_cls_name

            valset = pc_processor.dataset.semantic_kitti.SemanticKitti(
                root=self.settings.data_root,
                sequences=[8],
                config_path=data_config_path
            )
        else:
            raise ValueError(
                "invalid dataset: {}".format(self.settings.dataset))

        train_pv_loader = pc_processor.dataset.semanticKITTI_refine_PerspectiveViewLoader(
            dataset=trainset,
            config=self.settings.config,
            is_train=True, pcd_aug=False, img_aug=True, use_padding=True)

        val_pv_loader = pc_processor.dataset.semanticKITTI_refine_PerspectiveViewLoader(
            dataset=valset,
            config=self.settings.config,
            is_train=False, use_padding=True)

        train_loader = torch.utils.data.DataLoader(
            train_pv_loader,
            batch_size=self.settings.batch_size[0],
            num_workers=self.settings.n_threads,
            shuffle=True,
            drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            val_pv_loader,
            batch_size=self.settings.batch_size[1],
            num_workers=self.settings.n_threads,
            shuffle=False,
            drop_last=False
        )
        return train_loader, val_loader, None, None

    def _initCriterion(self):
        criterion = {}
        criterion["lovasz"] = pc_processor.loss.Lovasz_softmax(ignore=0)
        criterion["lovasz_point"] = pc_processor.loss.Lovasz_softmax_PointCloud(ignore=0)

        if self.settings.dataset == "SemanticKitti":
            alpha = np.log(1 + self.cls_weight)
            alpha = alpha / alpha.max()

        alpha[0] = 0
        if self.recorder is not None:
            self.recorder.logger.info("focal_loss alpha: {}".format(alpha))
        criterion["focal_loss"] = pc_processor.loss.FocalSoftmaxLoss(
            self.settings.nclasses, gamma=2, alpha=alpha, softmax=False)
        criterion["focal_loss_point"] = pc_processor.loss.FocalSoftmaxLoss_Point(
            self.settings.nclasses, gamma=2, alpha=alpha, softmax=False,ignore=0)

        # set device
        for _, v in criterion.items():
            v.cuda()
        return criterion

    def _backward(self, loss):
        self.optimizer.zero_grad()
        self.aux_optimizer.zero_grad()
        self.refine_optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()
        self.aux_optimizer.step()
        self.refine_optimizer.step()

    def _computeClassifyLoss(self, pred, label, label_mask):
        loss_foc = self.criterion["focal_loss"](pred, label, mask=label_mask)
        loss_lov = self.criterion["lovasz"](pred, label)
        return loss_lov, loss_foc

    def run(self, epoch, mode="Train"):
        if mode == "Train":
            dataloader = self.train_loader
            self.model.train()
            self.refine_module.train()
            if self.settings.distributed:
                self.train_sampler.set_epoch(epoch)

        elif mode == "Validation":
            dataloader = self.val_loader
            self.model.eval()
            self.refine_module.eval()
        else:
            raise ValueError("invalid mode: {}".format(mode))

        # init metrics meter
        loss_meter = pc_processor.utils.AverageMeter()
        loss_pcd_lovasz_meter = pc_processor.utils.AverageMeter()
        loss_pcd_focal_meter = pc_processor.utils.AverageMeter()
        loss_pcd = pc_processor.utils.AverageMeter()
        self.metrics.reset()

        loss_img_focal_meter = pc_processor.utils.AverageMeter()
        loss_img_lovasz_meter = pc_processor.utils.AverageMeter()
        entropy_img_meter = pc_processor.utils.AverageMeter()
        self.metrics_img.reset()

        total_iter = len(dataloader)
        t_start = time.time()

        feature_mean = torch.Tensor(self.settings.config["sensor"]["img_mean"]).unsqueeze(
            0).unsqueeze(2).unsqueeze(2).cuda()
        feature_std = torch.Tensor(self.settings.config["sensor"]["img_stds"]).unsqueeze(
            0).unsqueeze(2).unsqueeze(2).cuda()

        for i, (input_feature, input_mask, input_label, mapped_pointcloud, keep_poincloud, unproj_labels, n_points) in enumerate(dataloader):
            t_process_start = time.time()
            input_feature = input_feature.cuda()
            input_mask = input_mask.cuda()
            input_feature[:, 0:5] = (input_feature[:, 0:5] - feature_mean) / feature_std * \
                input_mask.unsqueeze(1).expand_as(input_feature[:, 0:5])
            pcd_feature = input_feature[:, 0:5]
            img_feature = input_feature[:, 5:8]

            input_label = input_label.cuda().long()
            label_mask = input_label.gt(0)

            if mode == "Train":
                lidar_pred, lidar_last_feature, camera_pred = self.model(pcd_feature, img_feature)

                tmp_pred = []
                tmp_labels = []
                for j in range(len(n_points)):
                    _npoints = n_points[j]
                    _px = mapped_pointcloud[j, :_npoints,0].to(torch.long)
                    _py = mapped_pointcloud[j, :_npoints,1].to(torch.long)
                    _unproj_labels = unproj_labels[j, :_npoints]
                    _points_xyz = keep_poincloud[j, :_npoints,:3]

                    # put in original pointcloud using indexes
                    _points_feature = lidar_last_feature[j, :, _px, _py]

                    # Filter out duplicate points
                    coords = np.round(_points_xyz[:, :3].cpu().numpy() / 0.05)
                    coords -= coords.min(0, keepdims=1)
                    coords, indices, inverse = sparse_quantize(coords, return_index=True,return_inverse=True)
                    coords = torch.tensor(coords, dtype=torch.int, device='cuda')

                    feats = _points_feature.permute(1, 0)[indices]
                    inputs = SparseTensor(coords=coords, feats=feats)
                    inputs = sparse_collate([inputs]).cuda()

                    _predict = self.refine_module(inputs)
                    _predict = _predict[inverse].permute(1, 0)

                    tmp_pred.append(_predict)
                    tmp_labels.append(_unproj_labels)

                predict = torch.cat(tmp_pred, -1).unsqueeze(0).to('cuda')
                unproj_labels = torch.cat(tmp_labels).unsqueeze(0).to('cuda')

                loss_lov_point = self.criterion["lovasz_point"](predict, unproj_labels)
                loss_foc_point = self.criterion["focal_loss_point"](predict, unproj_labels,mask=unproj_labels)
                loss_m = loss_foc_point + loss_lov_point

                loss_lov_cam, loss_foc_cam = self._computeClassifyLoss(
                    pred=camera_pred, label=input_label, label_mask=label_mask)

                total_loss = loss_m + loss_foc_cam + loss_lov_cam * self.settings.lambda_

                self._backward(total_loss)
                self.scheduler.step()
                self.refine_scheduler.step()
                self.aux_scheduler.step()
            else:
                with torch.no_grad():
                    lidar_pred, camera_pred = self.model(pcd_feature, img_feature)

                    tmp_pred = []
                    tmp_labels = []
                    for j in range(len(n_points)):
                        _npoints = n_points[j]
                        _px = mapped_pointcloud[j, :_npoints, 0].to(torch.long)
                        _py = mapped_pointcloud[j, :_npoints, 1].to(torch.long)
                        _unproj_labels = unproj_labels[j, :_npoints]
                        _points_xyz = keep_poincloud[j, :_npoints, :3]

                        # put in original pointcloud using indexes
                        _points_feature = lidar_pred[j, :, _px, _py]

                        coords = np.round(_points_xyz[:, :3].cpu().numpy() / 0.05)
                        coords -= coords.min(0, keepdims=1)
                        coords, indices, inverse = sparse_quantize(coords, return_index=True,return_inverse=True)
                        coords = torch.tensor(coords, dtype=torch.int, device='cuda')

                        feats = _points_feature.permute(1, 0)
                        feats = feats[indices]

                        inputs = SparseTensor(coords=coords, feats=feats)
                        inputs = sparse_collate([inputs]).cuda()

                        _predict = self.refine_module(inputs)
                        _predict = _predict[inverse].permute(1, 0)

                        tmp_pred.append(_predict)
                        tmp_labels.append(_unproj_labels)

                    predict = torch.cat(tmp_pred, -1).unsqueeze(0).to('cuda')
                    unproj_labels = torch.cat(tmp_labels).unsqueeze(0).to('cuda')

                    loss_lov_point = self.criterion["lovasz_point"](predict, unproj_labels)
                    loss_foc_point = self.criterion["focal_loss_point"](predict, unproj_labels,mask=unproj_labels)
                    loss_m = loss_lov_point + loss_foc_point

                    loss_lov_cam, loss_foc_cam = self._computeClassifyLoss(
                        pred=camera_pred, label=input_label, label_mask=label_mask)

                    total_loss = loss_m + loss_foc_cam + loss_lov_cam * self.settings.lambda_

                    if self.settings.n_gpus > 1:
                        total_loss = total_loss.mean()

            loss = total_loss.mean()

            with torch.no_grad():
                argmax = predict.argmax(dim=1)
                self.metrics.addBatch(argmax, unproj_labels)
                mean_iou_pcd, class_iou_pcd = self.metrics.getIoU()
                mean_acc_pcd, class_acc_pcd  = self.metrics.getAcc()
                mean_recall_pcd, class_recall_pcd = self.metrics.getRecall()

                argmax_img = camera_pred.argmax(dim=1)
                self.metrics_img.addBatch(argmax_img, input_label)
                mean_iou_img, class_iou_img = self.metrics_img.getIoU()
                mean_acc_img, class_acc_img = self.metrics_img.getAcc()
                mean_recall_img, class_recall_img = self.metrics_img.getRecall()

            loss_meter.update(total_loss.item(), input_feature.size(0))
            loss_pcd_lovasz_meter.update(loss_lov_point.item(), input_feature.size(0))
            loss_pcd_focal_meter.update(loss_foc_point.item(), input_feature.size(0))
            loss_pcd.update(loss_m.mean().item(), input_feature.size(0))

            loss_img_lovasz_meter.update(loss_lov_cam.item(), input_feature.size(0))
            loss_img_focal_meter.update(loss_foc_cam.item(), input_feature.size(0))

            # timer logger ----------------------------------------
            t_process_end = time.time()

            data_cost_time = t_process_start - t_start
            process_cost_time = t_process_end - t_process_start

            self.remain_time.update(cost_time=(time.time()-t_start), mode=mode)
            remain_time = datetime.timedelta(
                seconds=self.remain_time.getRemainTime(
                    epoch=epoch, iters=i, total_iter=total_iter, mode=mode
                ))
            t_start = time.time()

            if self.recorder is not None: #
                for g in self.refine_optimizer.param_groups:
                    lr = g["lr"]
                    break
                log_str = ">>> {} E[{:03d}|{:03d}] I[{:04d}|{:04d}] DT[{:.3f}] PT[{:.3f}] ".format(
                    mode, self.settings.n_epochs, epoch+1, total_iter, i+1, data_cost_time, process_cost_time)
                log_str += "LR {:0.5f} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} Recall {:0.4f} ".format(
                    lr, loss.item(), mean_acc_pcd.item(), mean_iou_pcd.item(), mean_recall_pcd.item())
                log_str += "ImgAcc {:0.4f} ImgIOU {:0.4F} ImgRecall {:0.4f}".format(
                    mean_acc_img.item(), mean_iou_img.item(), mean_recall_img.item())
                log_str += "RT {}".format(remain_time)
                self.recorder.logger.info(log_str)
            if self.settings.is_debug:
                break

        if self.recorder is not None:
            self.recorder.tensorboard.add_scalar(
                tag="{}_Loss".format(mode), scalar_value=loss_meter.avg, global_step=epoch)

            self.recorder.tensorboard.add_scalar(
                tag="{}_lr".format(mode), scalar_value=lr, global_step=epoch)

            self.recorder.tensorboard.add_scalar(
                tag="{}_meanAcc".format(mode), scalar_value=mean_acc_pcd.item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_meanIOU".format(mode), scalar_value=mean_iou_pcd.item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_meanRecall".format(mode), scalar_value=mean_recall_pcd.item(), global_step=epoch)

            for i, (_, v) in enumerate(self.mapped_cls_name.items()):
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_Acc".format(mode, i, v), scalar_value=class_acc_pcd[i].item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_Recall".format(mode, i, v), scalar_value=class_recall_pcd[i].item(),global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_IOU".format(mode, i, v), scalar_value=class_iou_pcd[i].item(), global_step=epoch)

            if epoch % self.settings.print_frequency == 0 :
                for i in range(pcd_feature.size(1)):
                    self.recorder.tensorboard.add_image(
                        "{}_PCDFeature_{}".format(mode, i), pcd_feature[0, i:i+1].cpu(), epoch)

                if camera_pred is not None:
                    for i in range(camera_pred.size(1)):
                        self.recorder.tensorboard.add_image(
                            "{}_RGBPred_cls_{:02d}_{}".format(mode, i, self.mapped_cls_name[i]), camera_pred[0, i:i+1].cpu(), epoch)

                for i in range(lidar_pred.size(1)):
                    self.recorder.tensorboard.add_image(
                        "{}_2DPred_cls_{:02d}_{}".format(mode, i, self.mapped_cls_name[i]), lidar_pred[0, i:i+1].cpu(), epoch)

                for i in range(lidar_pred.size(1)):
                    self.recorder.tensorboard.add_image("{}_2DLabel_cls_{:02d}_{}".format(
                        mode, i, self.mapped_cls_name[i]), input_label[0:1].eq(i).cpu(), epoch)

                self.recorder.tensorboard.add_image("{}_RGB".format(mode), img_feature[0].cpu(), epoch)

            log_str = ">>> {} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} Recall {:0.4f}".format(
                mode, loss_meter.avg, mean_acc_pcd.item(), mean_iou_pcd.item(), mean_recall_pcd.item())
            self.recorder.logger.info(log_str)

        result_metrics = {
            "Acc": mean_acc_pcd.item(),
            "IOU": mean_iou_pcd.item(),
            "Recall": mean_recall_pcd.item(),
            "last": 0
        }

        return result_metrics

