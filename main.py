from nerf_network import NeRF
from dataset import Dataset
from slice_renderer import SliceRenderer
import tqdm
import time
from datetime import date
import shutil
import csv

import torch
from torch.utils.tensorboard import SummaryWriter

import time
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NeUF:
    def __init__(self, **kwargs):
        self.grad_weight = kwargs.get("grad_weight", 0.1)
        self.seed = kwargs.get("seed",19981708)
        self.N_iters = kwargs.get("nb_iters_max",8500) #8500
        self.i_plot = kwargs.get("plot_freq",100) #100
        self.i_save = kwargs.get("save_freq",100)
        self.baked_dataset = kwargs.get("baked_dataset",True)
        self.training_mode = kwargs.get("training_mode","Random") #Slice, Random, RandomSpace...
        # self.training_mode = "Slice" #Slice, Random, RandomSpace...
        self.points_per_iter = kwargs.get("points_per_iter",50000) #used with random trainings
        self.jitter_training = False

        self.encoding = kwargs.get("encoding","None")

        self.datasetFolder = kwargs.get(
            "dataset",
            "D:\\0-Code\\NeUF\\data\\cerebral_data\\Pre_traitement_echo_v2\\Recalage\\Patient0\\us_recal_original\\baked_dataset.pkl",
        )


        self.ckptFile = kwargs.get("checkpoint","")
        # self.ckptFile = "latest\\ckpt.pkl"


        self.rootPoint = kwargs.get("root",".")

        ckpt = None
        self.dataset = None
        self.start = 0
        if self.ckptFile != "":
            ckpt = torch.load(self.ckptFile)

        if ckpt:
            print("Restarting from checkpoint:", self.ckptFile)
            np.random.seed(ckpt["seed"])
            torch.random.manual_seed(ckpt["seed"])
            self.seed = ckpt["seed"]

            if (ckpt.get("baked", False)):
                baked_dataset_path = ckpt.get("baked_dataset_file", ckpt.get("dataset_folder", self.datasetFolder))
                self.dataset = Dataset.open_from_save(baked_dataset_path)
            else:
                dataset_path = ckpt.get("dataset_folder", ckpt.get("baked_dataset_file", self.datasetFolder))
                self.dataset = Dataset(dataset_path)

            self.nerf = NeRF(ckpt)

            self.optimizer = torch.optim.Adam(params=self.nerf.grad_vars(), lr=5e-4, betas=(0.9, 0.999))
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            self.start = ckpt["start"]

        else:
            np.random.seed(self.seed)
            torch.random.manual_seed(self.seed)
            if self.baked_dataset:
                self.dataset = Dataset.open_from_save(self.datasetFolder)
            else:
                self.dataset = Dataset(self.datasetFolder)
            self.nerf = NeRF()


            if self.encoding == "Freq" :
                self.nerf.init_base_encoding(use_directions=False,
                                               use_encoding=True,
                                               num_freq=16,
                                               num_freq_dir=4)

            elif self.encoding == "Hash" :
                self.nerf.init_hash_encoding(bounding_box=self.dataset.get_bounding_box(),
                                             use_encoding=True,
                                             use_directions=False)

            elif self.encoding == "None" :
                self.nerf.init_hash_encoding(bounding_box=self.dataset.get_bounding_box(),
                                             use_encoding=False,
                                             use_directions=False)

            else :
                print("Unknown encoding: ", self.encoding)
                exit(-1)


            self.nerf.init_model(8, 256)
            self.optimizer = torch.optim.Adam(params=self.nerf.grad_vars(), lr=5e-4, betas=(0.9, 0.999))

        self.slice_renderer = SliceRenderer(self.dataset)

        d = date.today().strftime("%d-%m-%Y")
        logPath = f"{self.rootPoint}/logs/{d}"
        if not os.path.exists(logPath):
            os.makedirs(logPath)

        baseLogPath = logPath + "/" + self.nerf.get_rep_name() + "_" + self.getDatasetName()
        num = 0
        while os.path.exists(baseLogPath + "_" + str(num)):
            num += 1
        self.logPath = baseLogPath + "_" + str(num)
        os.makedirs(self.logPath, exist_ok=False)
        os.makedirs(self.logPath + "/checkpoints", exist_ok=False)
        os.makedirs(self.logPath + "/images", exist_ok=False)
        os.makedirs("./latest", exist_ok=True)
        
        self.gt_saved = False  # Flag to save ground truth images only once
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.logPath, "tensorboard"))
        self.loss_csv_path = os.path.join(self.logPath, "losses", "loss_history.csv")
        os.makedirs(os.path.join(self.logPath, "losses"), exist_ok=True)
        with open(self.loss_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "loss_train", "loss_valid", "loss_gt"])

    def run(self):

        t = time.time()
        start_time = time.time()
        loss_gt = None
        loss_valid = 0
        losses = []

        mse = torch.nn.MSELoss()
        # mse = torch.nn.L1Loss()
        
        # 梯度一致性损失的Scharr算子
        self.scharr_x = torch.tensor([
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 32.0
        
        self.scharr_y = torch.tensor([
            [-3, -10, -3],
            [0, 0, 0],
            [3, 10, 3]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 32.0
        
        self.scharr_x = self.scharr_x.to(device)
        self.scharr_y = self.scharr_y.to(device)
        
        i_min = -self.i_plot
        i_max = 0
        # generator = np.random.default_rng(self.seed)


        perm = torch.randperm(len(self.dataset.points))
        start_index = 0
        stop_index = self.points_per_iter
        indices = None

        self.precA1 = None
        self.precB1 = None
        self.precC1 = None
        self.precD1 = None
        A1 = None
        B1 = None
        C1 = None
        D1 = None

        progress_bar = tqdm.trange(self.start, self.N_iters + 1, desc="Training", dynamic_ncols=True)
        for i in progress_bar:
            # time.sleep(0.25)
            # print(i, end=" ")
            # if (self.i_plot >= 75):
                # self.progress.emit(i, i_min, i_max)
            if i % self.i_save == 0 and i != 0:
                params = self.nerf.get_save_dict()
                params.update({
                    "seed": self.seed,
                    "baked": self.baked_dataset,
                    "dataset_folder" if not self.baked_dataset else "baked_dataset_file": self.datasetFolder,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "start": i,
                    "bounding_box": self.dataset.get_bounding_box()
                })
                ckpt_fileName = self.logPath + "/checkpoints/" + str(i) + ".pkl"
                torch.save(params, ckpt_fileName)
                shutil.copy(ckpt_fileName,self.rootPoint+"/latest/ckpt.pkl")

            if i % self.i_plot == 0:
                # pbar.reset()

                i_min = i_max
                i_max += self.i_plot
                with torch.no_grad():
                    secs_per_iter = (time.time() - t) / self.i_plot
                    t = time.time()
                    total_time = t - start_time
                    time_str = str(datetime.timedelta(seconds=int(total_time)))

                    A1 = self.slice_renderer.render_slice_from_dataset_valid(self.nerf, 0,reshaped=True)
                    B1 = self.slice_renderer.render_slice_from_dataset_valid(self.nerf, 1,reshaped=True)
                    C1 = self.slice_renderer.render_slice_from_dataset_valid(self.nerf, 2,reshaped=True)
                    D1 = self.slice_renderer.render_slice_from_dataset_valid(self.nerf, 3,reshaped=True)

                    lossA1 = mse(A1, torch.reshape(self.dataset.get_slice_valid_pixels(0), (self.dataset.px_height, self.dataset.px_width)))
                    lossB1 = mse(B1, torch.reshape(self.dataset.get_slice_valid_pixels(1), (self.dataset.px_height, self.dataset.px_width)))
                    lossC1 = mse(C1, torch.reshape(self.dataset.get_slice_valid_pixels(2), (self.dataset.px_height, self.dataset.px_width)))
                    lossD1 = mse(D1, torch.reshape(self.dataset.get_slice_valid_pixels(3), (self.dataset.px_height, self.dataset.px_width)))

                    loss_valid = (lossA1 + lossC1 + lossB1 + lossD1) / 4
                    
                    # Save ground truth images only once
                    if not self.gt_saved:
                        gt_A1 = torch.reshape(self.dataset.get_slice_valid_pixels(0), (self.dataset.px_height, self.dataset.px_width))
                        gt_B1 = torch.reshape(self.dataset.get_slice_valid_pixels(1), (self.dataset.px_height, self.dataset.px_width))
                        gt_C1 = torch.reshape(self.dataset.get_slice_valid_pixels(2), (self.dataset.px_height, self.dataset.px_width))
                        gt_D1 = torch.reshape(self.dataset.get_slice_valid_pixels(3), (self.dataset.px_height, self.dataset.px_width))
                        
                        # Save raw tensors (cpu) for later inspection or re-use
                        torch.save(gt_A1.cpu(), self.logPath + "/images/A1_gt.pt")
                        torch.save(gt_B1.cpu(), self.logPath + "/images/B1_gt.pt")
                        torch.save(gt_C1.cpu(), self.logPath + "/images/C1_gt.pt")
                        torch.save(gt_D1.cpu(), self.logPath + "/images/D1_gt.pt")
                        
                        # Save quick PNG visualizations (grayscale)
                        plt.imsave(self.logPath + "/images/A1_gt.png", gt_A1.cpu().numpy(), cmap='gray')
                        plt.imsave(self.logPath + "/images/B1_gt.png", gt_B1.cpu().numpy(), cmap='gray')
                        plt.imsave(self.logPath + "/images/C1_gt.png", gt_C1.cpu().numpy(), cmap='gray')
                        plt.imsave(self.logPath + "/images/D1_gt.png", gt_D1.cpu().numpy(), cmap='gray')
                        
                        self.gt_saved = True
                        tqdm.tqdm.write(f"Ground truth images saved to {self.logPath}/images/")

                    if self.precA1 is not None and self.precB1 is not None and self.precC1 is not None and self.precD1 is not None:
                        varA1 = (torch.sum(torch.square(torch.sub(self.precA1,A1))))
                        varB1 = (torch.sum(torch.square(torch.sub(self.precB1,B1))))
                        varC1 = (torch.sum(torch.square(torch.sub(self.precC1,C1))))
                        varD1 = (torch.sum(torch.square(torch.sub(self.precD1,D1))))

                        loss_gt = (varA1 + varC1 + varB1 + varD1) / 4

                    params = {
                        "A1": torch.reshape(A1, (self.dataset.px_height, self.dataset.px_width)),
                        "B1": torch.reshape(B1, (self.dataset.px_height, self.dataset.px_width)),
                        "C1": torch.reshape(C1, (self.dataset.px_height, self.dataset.px_width)),
                        "D1": torch.reshape(D1, (self.dataset.px_height, self.dataset.px_width)),
                        "iteration": i,
                        "time": time_str,
                        "loss_train": float(np.mean(losses)) if losses else (float(loss_gt) if loss_gt is not None else None),
                        "loss_valid": float(loss_valid.detach().cpu()),
                        "loss_gt": float(loss_gt.detach().cpu()) if loss_gt is not None else None,
                        "i_plot": self.i_plot
                    }
                    # self.new_values.emit(params)
                    # Save preview images and raw tensors to the run folder so training
                    # records keep a history of rendered slices for this iteration.
                    img_dir = os.path.join(self.logPath, "images")
                    try:
                        # save raw tensors (cpu) for later inspection or re-use
                        torch.save(A1.cpu(), os.path.join(img_dir, f"A1_{i}.pt"))
                        torch.save(B1.cpu(), os.path.join(img_dir, f"B1_{i}.pt"))
                        torch.save(C1.cpu(), os.path.join(img_dir, f"C1_{i}.pt"))
                        torch.save(D1.cpu(), os.path.join(img_dir, f"D1_{i}.pt"))

                        # save quick PNG visualizations (grayscale)
                        plt.imsave(os.path.join(img_dir, f"A1_{i}.png"), A1.cpu().numpy(), cmap='gray')
                        plt.imsave(os.path.join(img_dir, f"B1_{i}.png"), B1.cpu().numpy(), cmap='gray')
                        plt.imsave(os.path.join(img_dir, f"C1_{i}.png"), C1.cpu().numpy(), cmap='gray')
                        plt.imsave(os.path.join(img_dir, f"D1_{i}.png"), D1.cpu().numpy(), cmap='gray')
                    except Exception as e:
                        # Don't stop training if saving fails; just warn
                        print("Warning: failed to save preview images/tensors:", e)

                    if params["loss_train"] is not None:
                        self.tb_writer.add_scalar("loss/train", params["loss_train"], i)
                    if params["loss_valid"] is not None:
                        self.tb_writer.add_scalar("loss/valid", params["loss_valid"], i)
                    if params["loss_gt"] is not None:
                        self.tb_writer.add_scalar("loss/gt", params["loss_gt"], i)
                    with open(self.loss_csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            i,
                            params["loss_train"],
                            params["loss_valid"],
                            params["loss_gt"],
                        ])
                    self.tb_writer.flush()
                    progress_bar.set_postfix(
                        secs_per_iter=f"{secs_per_iter:.4f}",
                        loss_train=f"{params['loss_train']:.6f}" if params["loss_train"] is not None else "None",
                        loss_valid=f"{params['loss_valid']:.6f}" if params["loss_valid"] is not None else "None",
                        loss_gt=f"{params['loss_gt']:.6f}" if params["loss_gt"] is not None else "None",
                        elapsed=time_str,
                    )

            self.precA1 = A1
            self.precB1 = B1
            self.precC1 = C1
            self.precD1 = D1

            if self.training_mode == "Slice":
                img_i = np.random.randint(len(self.dataset.slices))
                # target = self.dataset.slices[img_i].image.to(device)
                target = self.dataset.get_slice_pixels(img_i).to(device)
                density = self.slice_renderer.render_slice_from_dataset(self.nerf, img_i, jitter=self.jitter_training)

            elif self.training_mode == "Random":
                # indices = generator.choice(len(self.dataset.points), self.points_per_iter, replace=False)
                if(stop_index > start_index) :
                    indices = perm[start_index:stop_index]
                else :
                    indices = perm[start_index:-1]
                    indices = torch.cat((indices,perm[0:stop_index]))
                start_index = stop_index
                stop_index = (start_index+self.points_per_iter)

                if(stop_index >= len(self.dataset.points)) :
                    stop_index = stop_index - len(self.dataset.points) + 1

                target = self.dataset.get_indices_pixels(indices)
                density = self.slice_renderer.query_random_positions(self.nerf,indices,reshaped=False)
            else :
                print("Training mode: ", self.training_mode, "not recognized or implemented")
                exit(-1)

            self.optimizer.zero_grad()

            loss = mse(density, target)
            
            # 添加梯度一致性损失
            if self.training_mode == "Slice":
                # 对Slice模式，计算整个切片的梯度一致性
                target_slice = torch.reshape(target, (self.dataset.px_height, self.dataset.px_width)).unsqueeze(0).unsqueeze(0)
                density_slice = torch.reshape(density, (self.dataset.px_height, self.dataset.px_width)).unsqueeze(0).unsqueeze(0)
                
                # 计算梯度
                target_grad_x = torch.nn.functional.conv2d(target_slice, self.scharr_x, padding=1)
                target_grad_y = torch.nn.functional.conv2d(target_slice, self.scharr_y, padding=1)
                
                density_grad_x = torch.nn.functional.conv2d(density_slice, self.scharr_x, padding=1)
                density_grad_y = torch.nn.functional.conv2d(density_slice, self.scharr_y, padding=1)
                
                # 梯度一致性损失
                grad_loss = mse(density_grad_x, target_grad_x) + mse(density_grad_y, target_grad_y)
                loss = loss + self.grad_weight * grad_loss
            
            loss.backward()
            self.optimizer.step()

            losses.append(loss.detach().cpu())

        self.tb_writer.close()
        progress_bar.close()

    def getReferences(self):
        return (torch.reshape(self.dataset.get_slice_valid_pixels(0), (self.dataset.px_height, self.dataset.px_width)),
                torch.reshape(self.dataset.get_slice_valid_pixels(1), (self.dataset.px_height, self.dataset.px_width)),
                torch.reshape(self.dataset.get_slice_valid_pixels(2), (self.dataset.px_height, self.dataset.px_width)),
                torch.reshape(self.dataset.get_slice_valid_pixels(3), (self.dataset.px_height, self.dataset.px_width)))

    def getGT(self):
        if self.dataset.has_gt :
            return (torch.reshape(self.dataset.get_slice_valid_gt(0), (self.dataset.px_height, self.dataset.px_width)),
                    torch.reshape(self.dataset.get_slice_valid_gt(1), (self.dataset.px_height, self.dataset.px_width)),
                    torch.reshape(self.dataset.get_slice_valid_gt(2), (self.dataset.px_height, self.dataset.px_width)),
                    torch.reshape(self.dataset.get_slice_valid_gt(3), (self.dataset.px_height, self.dataset.px_width)))
        return (None,None,None,None)

    def getEncodingName(self):
        return self.nerf.get_encode_name()

    def getDatasetName(self):
        return self.dataset.name


if __name__ == "__main__":
    neuf = NeUF(encoding = "Hash")
    neuf.run()
