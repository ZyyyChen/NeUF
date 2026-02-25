from nerf_network import NeRF
from dataset import Dataset
from slice_renderer import SliceRenderer
import tqdm
import time
from datetime import date
import shutil

import torch

import time
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

# from PySide6.QtCore import QObject, Signal

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NeUF():
    # progress = Signal(int, int, int)
    # new_values = Signal(dict)

    def __init__(self, **kwargs):
        super().__init__()

        self.seed = kwargs.get("seed",19981708)
        self.N_iters = kwargs.get("nb_iters_max",100000)
        self.i_plot = kwargs.get("plot_freq",500)
        self.i_save = kwargs.get("save_freq",500)
        self.baked_dataset = kwargs.get("baked_dataset",True)
        self.training_mode = kwargs.get("training_mode","Random") #Slice, Random, RandomSpace...
        # self.training_mode = "Slice" #Slice, Random, RandomSpace...
        self.points_per_iter = kwargs.get("points_per_iter",50000) #used with random trainings
        self.jitter_training = False

        self.encoding = kwargs.get("encoding","None")

        # self.datasetFolder = "E:/NeRF-3/datasets/baked/seinGrandeSphereValidNew.pkl"
        # self.datasetFolder = "E:/NeRF-3/datasets/baked/dicomBalayageTotalValidNew.pkl"
        # self.datasetFolder = "E:/NeRF-3/datasets/baked/test/blue.pkl"
        self.datasetFolder = kwargs.get("dataset","E:/NeRF-3/datasets/baked/test/cube.pkl")
        # self.datasetFolder = "E:/NeRF-3/datasets/baked/test/dicom.pkl"
        # self.datasetFolder = "E:/NeRF-3/datasets/baked/CubeBalayageTotalValidNew.pkl"
        # self.datasetFolder = "E:/NeRF-3/datasets/baked/sphereTotalValid.pkl"
        # self.datasetFolder = "E:/NeRF-3/datasets/baked/Vass2.pkl"
        # self.datasetFolder = "E:/NeRF-3/datasets/baked/GTMRI.pkl"
        # self.datasetFolder = "E:/NeRF-3/datasets/baked/exportsimple.pkl"
        # self.datasetFolder = "E:/NeRF-3/datasets/baked/tube.pkl"
        # self.datasetFolder = "E:/NeRF-3/datasets/baked/seinGrandeFixed.pkl"

        self.ckptFile = kwargs.get("checkpoint","")
        # ckptFile = "logs/09-05-2023_HASH_0/checkpoints/11000.pkl"
        # ckptFile = "logs/12-05-2023_NONE_1/checkpoints/13000.pkl"

        self.rootPoint = kwargs.get("root","C:/Users/anchling/Documents/projects/neural-ultrasound-field")

        ckpt = None
        self.dataset = None
        self.start = 0
        if self.ckptFile != "":
            ckpt = torch.load(self.ckptFile, map_location=device)

        if ckpt:
            print("Restarting from checkpoint:", self.ckptFile)
            np.random.seed(ckpt["seed"])
            torch.random.manual_seed(ckpt["seed"])
            self.seed = ckpt["seed"]

            if (ckpt.get("baked", False)):
                self.dataset = Dataset.open_from_save(ckpt["baked_dataset_file"])
            else:
                self.dataset = Dataset(ckpt["dataset_folder"])

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
                                               num_freq=10,
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
        logPath = os.path.join(self.rootPoint,'logs',d)
        os.makedirs(logPath, exist_ok=True)

        baseLogPath = logPath + "/" + self.nerf.get_rep_name() + "_" + self.getDatasetName()
        num = 0
        while os.path.exists(baseLogPath + "_" + str(num)):
            num += 1
        self.logPath = baseLogPath + "_" + str(num)
        os.mkdir(self.logPath)
        os.mkdir(self.logPath + "/checkpoints")
        os.mkdir(self.logPath + "/images")
        os.mkdir(self.logPath + "/losses")

    def run(self):

        t = time.time()
        start_time = time.time()
        loss_gt = None
        loss_valid = 0
        losses = []

        mse = torch.nn.MSELoss()
        # mse = torch.nn.L1Loss()
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
        A3 = None
        B3 = None
        C3 = None
        D3 = None

        for i in range(self.start, self.N_iters + 1):
            # time.sleep(0.25)
            # print(i, end=" ")
            # if (self.i_plot >= 75):
            #     self.progress.emit(i, i_min, i_max)
            if i % self.i_save == 0 and i != 0:
                params = self.nerf.get_save_dict()
                params.update({
                    "seed": self.seed,
                    "baked": self.baked_dataset,
                    "dataset_folder" if not self.baked_dataset else "baked_dataset_file": self.datasetFolder,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "start": i,
                    "elapsed":int(time.time() - start_time),
                    "bounding_box": self.dataset.get_bounding_box(),
                    "dims2D": (self.dataset.orig_px_width, self.dataset.orig_px_height), 
                    "roi2D": self.dataset.roi_2d,
                })
                ckpt_fileName = self.logPath + "/checkpoints/" + str(i) + ".pkl"
                torch.save(params, ckpt_fileName)
                shutil.copy(ckpt_fileName,self.rootPoint+"/latest/ckpt.pkl")

            if i % self.i_plot == 0:
                # pbar.reset()

                i_min = i_max
                i_max += self.i_plot
                with torch.no_grad():
                    print("\n________________________\n", (time.time() - t) / self.i_plot,
                          'secs per iter. Result obtained at',
                          datetime.datetime.now(), '\n______________________\n')
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

                    if self.precA1 is not None and self.precB1 is not None and self.precC1 is not None and self.precD1 is not None:
                        varA1 = (torch.sum(torch.square(torch.sub(self.precA1,A1))))
                        varB1 = (torch.sum(torch.square(torch.sub(self.precB1,B1))))
                        varC1 = (torch.sum(torch.square(torch.sub(self.precC1,C1))))
                        varD1 = (torch.sum(torch.square(torch.sub(self.precD1,D1))))

                        loss_gt = (varA1 + varC1 + varB1 + varD1) / 4

                    # if self.dataset.has_gt:
                    #     loss_gtA1 = mse(A1, torch.reshape(self.dataset.get_slice_valid_gt(0), (self.dataset.px_height, self.dataset.px_width)))
                    #     loss_gtB1 = mse(B1, torch.reshape(self.dataset.get_slice_valid_gt(0), (self.dataset.px_height, self.dataset.px_width)))
                    #     loss_gtC1 = mse(C1, torch.reshape(self.dataset.get_slice_valid_gt(0), (self.dataset.px_height, self.dataset.px_width)))
                    #     loss_gtD1 = mse(D1, torch.reshape(self.dataset.get_slice_valid_gt(0), (self.dataset.px_height, self.dataset.px_width)))
                    #
                    #     loss_gt = (loss_gtA1 + loss_gtC1 + loss_gtB1 + loss_gtD1) / 4
                    #
                    # else:
                    #     loss_gt = None


                    params = {
                        "A1": torch.reshape(A1, (self.dataset.px_height, self.dataset.px_width)),
                        "B1": torch.reshape(B1, (self.dataset.px_height, self.dataset.px_width)),
                        "C1": torch.reshape(C1, (self.dataset.px_height, self.dataset.px_width)),
                        "D1": torch.reshape(D1, (self.dataset.px_height, self.dataset.px_width)),
                        "iteration": i,
                        "time": time_str,
                        "loss_train": np.mean(losses) if losses else loss_gt,
                        # "loss_valid": loss_valid.cpu().numpy(),
                        "loss_valid": 4,
                        # "loss_gt": loss_gt,
                        "loss_gt": 4,
                        "i_plot": self.i_plot
                    }
                    # self.new_values.emit(params)

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
            loss.backward()
            self.optimizer.step()

            losses.append(loss.detach().cpu())

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
    neuf = NeUF(encoding="Freq",)
    neuf.run()
