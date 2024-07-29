import argparse
import datetime
from option import Option
import os
import torch
import time
import yaml
import trainer
import pc_processor

class Experiment(object):
    def __init__(self, settings: Option):
        self.settings = settings
        os.environ["CUDA_VISIBLE_DEVICES"] = self.settings.gpu

        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed(self.settings.seed)
        torch.cuda.set_device(int(self.settings.gpu))
        torch.backends.cudnn.benchmark = True

        if not self.settings.distributed or (self.settings.rank == 0):
            self.recorder = pc_processor.checkpoint.Recorder(
                self.settings, self.settings.save_path)
        else:
            self.recorder = None

        self.epoch_start = 0
        # init model
        self.model = pc_processor.models.DBNet(
            pcd_channels=5,
            img_channels=3,
            nclasses=self.settings.nclasses,
            base_channels=self.settings.base_channels,
            image_backbone=self.settings.img_backbone,
            imagenet_pretrained=self.settings.imagenet_pretrained,
        )

        self.trainer = trainer.Trainer(
            self.settings, self.model, self.recorder)

        self._loadCheckpoint()

    def _loadCheckpoint(self):
        assert self.settings.pretrained_model is None or self.settings.checkpoint is None, "cannot use pretrained weight and checkpoint at the same time"
        if self.settings.pretrained_model is not None:
            if not os.path.isfile(self.settings.pretrained_model):
                raise FileNotFoundError("pretrained model not found: {}".format(self.settings.pretrained_model))
            state_dict = torch.load(self.settings.pretrained_model, map_location="cuda:0")
            new_state_dict = self.model.state_dict()
            for k, v in state_dict['model'].items():
                if k in new_state_dict.keys():
                    if new_state_dict[k].size() == v.size():
                        new_state_dict[k] = v
                    else:
                        print("diff size: ", k, v.size())
                else:
                    print("diff key: ", k)
            self.model.load_state_dict(new_state_dict)

            for param in self.model.parameters():
                param.requires_grad = False

            if self.recorder is not None:
                self.recorder.logger.info(
                    "loading pretrained weight from: {}".format(self.settings.pretrained_model))

        if self.settings.checkpoint is not None:
            if not os.path.isfile(self.settings.checkpoint):
                raise FileNotFoundError(
                    "checkpoint file not found: {}".format(self.settings.checkpoint))
            checkpoint_data = torch.load(
                self.settings.checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint_data["model"])
            self.trainer.optimizer.load_state_dict(
                checkpoint_data["optimizer"])
            self.trainer.aux_optimizer.load_state_dict(
                checkpoint_data["aux_optimizer"])
            self.trainer.refine_optimizer.load_state_dict(checkpoint_data["refine_optimizer"])
            self.epoch_start = checkpoint_data["epoch"] + 1

    def run(self):
        t_start = time.time()
        if self.settings.val_only:
            self.trainer.run(0, mode="Validation")
            return
        
        best_val_result = None

        for epoch in range(self.epoch_start, self.settings.n_epochs):
            self.trainer.run(epoch, mode="Train")

            if epoch % self.settings.val_frequency == 0 or epoch == self.settings.n_epochs-1:
                val_result = self.trainer.run(epoch, mode="Validation")
                if self.recorder is not None:
                    if best_val_result is None:
                        best_val_result = val_result
                    for k, v in val_result.items():
                        if v >= best_val_result[k]:
                            self.recorder.logger.info("get better {} model: {}".format(k, v))
                            saved_path = os.path.join(self.recorder.checkpoint_path, "best_{}_model.pth".format(k))
                            best_val_result[k] = v
                            state = {
                                "model": self.model.state_dict(),
                                "refine_module": self.trainer.refine_module.state_dict(),
                                "optimizer": self.trainer.optimizer.state_dict(),
                                "aux_optimizer": self.trainer.aux_optimizer.state_dict(),
                                "refine_optimizer": self.trainer.refine_optimizer.state_dict(),
                                "epoch": epoch,
                            }
                            torch.save(state, saved_path)

            # save checkpoint
            if self.recorder is not None:
                saved_path = os.path.join(self.recorder.checkpoint_path, "checkpoint.pth")
                checkpoint_data = {
                    "model": self.model.state_dict(),
                    "refine_module":self.trainer.refine_module.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "aux_optimizer": self.trainer.aux_optimizer.state_dict(),
                    "refine_optimizer":self.trainer.refine_optimizer.state_dict(),
                    "epoch": epoch,
                }

                torch.save(checkpoint_data, saved_path)
                # log
                if best_val_result is not None:
                    log_str = ">>> Best Result: "
                    for k, v in best_val_result.items():
                        log_str += "{}: {} ".format(k, v)
                    self.recorder.logger.info(log_str)
        cost_time = time.time() - t_start
        if self.recorder is not None:
            self.recorder.logger.info("==== total cost time: {}".format(
                datetime.timedelta(seconds=cost_time)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Options")
    parser.add_argument("--config_path", type=str, metavar="config_path",
                        help="path of config file, type: string", default='tasks/VPA/config_server_kitti.yaml')
    parser.add_argument("--id", type=int, metavar="experiment_id", required=False,
                        help="id of experiment", default=0)
    args = parser.parse_args()
    exp = Experiment(Option(args.config_path))
    print("===init env success===")
    exp.run()
