import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import crop
import numpy as np
from utils import get_base_points, get_oriented_points_and_views
from tqdm import tqdm
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Dataset:
    def __init__(self, folder, nb_valid=4, seed=-1, **kwargs):
        self.width = 0
        self.height = 0
        self.px_width = 0
        self.px_height = 0
        self.orig_px_width = 0
        self.orig_px_height = 0
        self.point_min = (0,0,0)
        self.point_max = (100,100,100)
        self.slices = []
        self.slices_valid = []
        self.X, self.Y = ([],[])
        self.name = kwargs.get("name", os.path.basename(folder)) 
        self.has_gt = False
        self.roi_2d = None
        self.image_step = kwargs.get("image_step", 1)

        img_folder = kwargs.get("img_folder","us")
        info_folder = kwargs.get("info_folder", "")  
        prefix = kwargs.get("prefix","us/img_")
        suffix = kwargs.get("suffix",".jpg")
        reverse_quat = kwargs.get("reverse_quat",False) #TODO: reverse_quat?
        self.exclude_valid = kwargs.get("exclude_valid", True)


        image_buffer = []
        gt_buffer = []
        points_buffer = []
        points_numpy = []
        viewdirs_buffer = []
        image_valid_buffer = []
        gt_valid_buffer = []
        points_valid_buffer = []
        viewdirs_valid_buffer = []

        pos_buffer = []
        rot_buffer = []

        seed = 17081998
        if seed != -1 :
            np.random.seed(seed)
            torch.random.manual_seed(seed)

        roi_json_path = os.path.join(folder, "..", "..", "bounding_box.json")
        if os.path.exists(roi_json_path):
            with open(roi_json_path, 'r') as f:
                data = json.load(f)["bounding_box"]
                self.roi_2d = data if data["width"] > 1 else None

        first_image = torch.squeeze( read_image(os.path.join(folder, img_folder, prefix + "0" + suffix), ImageReadMode.GRAY) )
        self.orig_px_height, self.orig_px_width = first_image.shape
        
        with open(os.path.join(folder, info_folder,"infos.json"), "r") as f:
            infos = json.load(f)["infos"]
            if "scan_dims_mm" in infos:
                self.width = float(infos["scan_dims_mm"]["width"])
                self.height = float(infos["scan_dims_mm"]["depth"])

        with open(os.path.join(folder, info_folder,"infos.dat"), "r") as info:
            lines = info.readlines()
            nb_line = 0
            for i in tqdm(range(0,len(lines),self.image_step),desc="Opening dataset", total=int(len(lines)/self.image_step)):
                l = lines[i]
                if l and l[0] == "D":
                    dimensions = l[2:-1].split(" ")
                    self.point_min = (float(dimensions[0]), float(dimensions[1]), float(dimensions[2]))
                    self.point_max = (float(dimensions[3]), float(dimensions[4]), float(dimensions[5]))
                    nb_line += 1
                elif l and l[0] == "m":
                    splits, dimensions = l[2:-1].split(" "), []
                    for element in splits:
                        try :
                            dimensions.append(float("".join([ele for ele in element if ele.isdigit() or ele == "." or ele == "-"])))
                        except ValueError :
                            pass
                    self.point_min = (float(dimensions[0]), float(dimensions[1]), float(dimensions[2]))
                    self.point_max = (float(dimensions[3]), float(dimensions[4]), float(dimensions[5]))
                else :
                    inf = l.split('\n')[0].split(" ")
        
                    pos = np.array(list(map(float, inf[:3])))

                    if reverse_quat :
                        quat = Quat(float(inf[6]), float(inf[3]), float(inf[4]), float(inf[5]))
                    else :
                        quat = Quat(float(inf[3]), -float(inf[4]), -float(inf[5]), -float(inf[6]))

                    img_name = os.path.join(folder, img_folder, prefix + str(nb_line) + suffix)
                    gt_name = " "
                    image = self.get_torch_image(img_name)
                    if os.path.exists(gt_name) :
                        self.has_gt = True
                        gt = self.get_torch_image(gt_name)
                    elif not self.has_gt :
                        gt = None
                    else :
                        print("all or none ground truth must be provided")
                        exit(-1)

                    if (
                        # (self.width and width != self.width) or
                        #     (self.height and height != self.height) or
                            (self.px_width and self.px_width != image.shape[1]) or
                            (self.px_height and self.px_height != image.shape[0])):
                        print("images must be of consistent dimensions mm and px")
                        exit(-1)

                    self.px_width = image.shape[1]
                    self.px_height = image.shape[0]

                    self.X, self.Y = ([],[])
                    if self.X == [] and self.Y == []:
                        self.X, self.Y = get_base_points(
                            self.width, 
                            self.height, 
                            self.px_width, 
                            self.px_height,
                            offset_x_mm=getattr(self, 'roi_offset_x_mm', 0),
                            offset_y_mm=getattr(self, 'roi_offset_y_mm', 0)
)

                    image_buffer.append(torch.squeeze(torch.reshape(image,(1,-1))))
                    if self.has_gt:
                        gt_buffer.append(torch.squeeze(torch.reshape(gt,(1,-1))))

                    p,v = get_oriented_points_and_views(self.X,self.Y, pos, quat)

                    points_buffer.append(torch.from_numpy(p.astype(dtype=np.float32)).to(device))
                    viewdirs_buffer.append(torch.from_numpy(v.astype(dtype=np.float32)).to(device))
                    
                    points_numpy.append(p)

                    pos_buffer.append(pos)
                    rot_buffer.append(quat)

                    nb_line += self.image_step

                l = info.readline()

        # Find the correct points min and max:
        points_numpy = np.array(points_numpy)
        self.point_min = np.min([np.min(p,axis=0) for p in points_numpy],axis=0)
        self.point_max = np.max([np.max(p,axis=0) for p in points_numpy],axis=0)

        self.point_min_dev = torch.FloatTensor(self.point_min).to(device)
        self.point_max_dev = torch.FloatTensor(self.point_max).to(device)
        self.nb_valid = nb_valid

        i_valid = np.random.choice(range(int(nb_line/self.image_step)), nb_valid, replace = False)

        j = 0
        k = 0
        nb_images = len(image_buffer)
        for i in range(nb_images-1,0,-1) :
            if i in i_valid :
                self.slices_valid.append(Slice(j*self.px_width*self.px_height,(j+1)*self.px_width*self.px_height, pos_buffer[i], rot_buffer[i]))

                if self.exclude_valid :
                    image_valid_buffer.append(image_buffer.pop(i))
                    if self.has_gt :
                        gt_valid_buffer.append(gt_buffer.pop(i))
                    points_valid_buffer.append(points_buffer.pop(i))
                    viewdirs_valid_buffer.append(viewdirs_buffer.pop(i))
                    j += 1
                else :
                    #Also include the slices in the base dataset
                    self.slices.append(
                        Slice(k * self.px_width * self.px_height, (k + 1) * self.px_width * self.px_height,
                              pos_buffer[i], rot_buffer[i]))
                    k += 1

                    image_valid_buffer.append(image_buffer[i])
                    if self.has_gt:
                        gt_valid_buffer.append(gt_buffer[i])
                    points_valid_buffer.append(points_buffer[i])
                    viewdirs_valid_buffer.append(viewdirs_buffer[i])
                    j += 1
            else :
                self.slices.append(Slice(k*self.px_width*self.px_height,(k+1)*self.px_width*self.px_height, pos_buffer[i], rot_buffer[i]))
                k+=1

        self.pixels = torch.flatten(torch.stack(image_buffer))
        self.pixels_valid = torch.flatten(torch.stack(image_valid_buffer))
        if self.has_gt:
            self.gt = torch.flatten(torch.stack(gt_buffer))
            self.gt_valid = torch.flatten(torch.stack(gt_valid_buffer))

        self.points = torch.cat(points_buffer)
        self.points_valid = torch.cat(points_valid_buffer)
        self.viewdirs = torch.cat(viewdirs_buffer)
        self.viewdirs_valid = torch.cat(viewdirs_valid_buffer)

        print("\nopened dataset", folder, "containing", nb_images, "slices\ndimensions:\n\tmin:",self.point_min,"\n\tmax",self.point_max)

    def get_torch_image(self, img_path: str,) -> torch.Tensor:
        """
        Loads a grayscale image as a torch tensor, optionally cropping using a given ROI.
        
        Note: Cropping happens on CPU before moving to GPU (more efficient)
        """
        # 1. 在CPU上读取图像
        image = read_image(img_path, ImageReadMode.GRAY)  # shape: [1, H, W]

        # 2. 在CPU上进行裁剪（内存操作，不需要GPU）
        if self.roi_2d:
            required_keys = {'x', 'y', 'width', 'height'}
            if not required_keys.issubset(self.roi_2d.keys()):
                raise ValueError(f"ROI must contain keys {required_keys}")
            # crop(img, top, left, height, width)
            image = crop(image, self.roi_2d['y'], self.roi_2d['x'], 
                        self.roi_2d['height'], self.roi_2d['width'])

        # 3. flip the image horizontally to match the physical orientation
        # image = torch.flip(image, [2])

        # 4. 最后才转到GPU（只转一次，数据量最小）
        return torch.squeeze(image.float()).to(device)
        

    def get_bounding_box(self):
        ret = (self.point_min_dev, self.point_max_dev)
        return ret

    def save(self,file_name):
        dic = {
            "dataset":self
        }
        os.makedirs(os.path.dirname(file_name),exist_ok=True)
        torch.save(dic,file_name)

    def get_slice_pixels(self,number):
        return torch.unsqueeze(self.pixels[self.slices[number].start:self.slices[number].end],1)

    def get_slice_valid_pixels(self,number):
        return torch.unsqueeze(self.pixels_valid[self.slices_valid[number].start:self.slices_valid[number].end],1)

    def get_slice_gt(self,number):
        if self.has_gt :
            return torch.unsqueeze(self.gt[self.slices[number].start:self.slices[number].end],1)
        return None

    def get_slice_valid_gt(self,number):
        if self.has_gt :
            return torch.unsqueeze(self.gt_valid[self.slices_valid[number].start:self.slices_valid[number].end],1)
        return None
    def get_slice_points(self,number):
        return torch.unsqueeze(self.points[self.slices[number].start:self.slices[number].end],1)

    def get_slice_valid_points(self,number):
        return torch.unsqueeze(self.points_valid[self.slices_valid[number].start:self.slices_valid[number].end],1)

    def get_slice_viewdirs(self,number):
        return torch.unsqueeze(self.viewdirs[self.slices[number].start:self.slices[number].end],1)

    def get_slice_valid_viewdirs(self,number):
        return torch.unsqueeze(self.viewdirs_valid[self.slices_valid[number].start:self.slices_valid[number].end],1)

    def get_indices_pixels(self,indexes):
        return torch.unsqueeze(self.pixels[indexes],1)

    def get_indices_pixels_valid(self,indexes):
        return torch.unsqueeze(self.pixels_valid[indexes],1)
    def get_indices_points(self,indexes):
        return torch.unsqueeze(self.points[indexes],1)

    def get_indices_points_values(self,indexes):
        return torch.unsqueeze(self.points_valid[indexes],1)


    @staticmethod
    def open_from_save(save_file):
        save = torch.load(save_file, weights_only=False, map_location=device)
        dataset = save["dataset"]

        return dataset


class Slice:
    def __init__(self, start, end, position, rotation):
        self.position = position
        self.rotation = rotation
        self.start = int(start)
        self.end = int(end)



class Quat:
    def __init__(self, w, x, y, z):

        self.w = w
        self.x = x
        self.y = y
        self.z = z

        self.compute_quat_params()

    # normalize the quaternion
    def normalize(self):
        norm = np.sqrt(self.qw2 + self.qx2 + self.qy2 + self.qz2)
        self.w /= norm
        self.x /= norm
        self.y /= norm
        self.z /= norm

        self.compute_quat_params()

    def compute_quat_params(self):

        self.qw2 = self.w ** 2  # 0
        self.qx2 = self.x ** 2  # 1
        self.qy2 = self.y ** 2  # 2
        self.qz2 = self.z ** 2  # 3

        self.dqxqy = self.x * self.y * 2  # 4
        self.dqwqz = self.w * self.z * 2  # 5
        self.dqxqz = self.x * self.z * 2  # 6
        self.dqwqy = self.w * self.y * 2  # 7
        self.dqyqz = self.y * self.z * 2  # 8
        self.dqwqx = self.w * self.x * 2  # 9

    def apply_quat(self, point):
        return np.array([
            point[0] * (self.qw2 + self.qx2 - self.qy2 - self.qz2) + point[1] * (self.dqxqy + self.dqwqz) + point[2] * (
                        self.dqxqz - self.dqwqy),

            point[0] * (self.dqxqy - self.dqwqz) + point[1] * (self.qw2 - self.qx2 + self.qy2 - self.qz2) + point[2] * (
                        self.dqyqz + self.dqwqx),

            point[0] * (self.dqxqz + self.dqwqy) + point[1] * (self.dqyqz - self.dqwqx) + point[2] * (
                        self.qw2 - self.qx2 - self.qy2 + self.qz2)
        ])
    
    def as_rotmat(self):
        # Returns a 3x3 rotation matrix
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])

    def __repr__(self):
        return str(self.qw2**0.5) + "_" + str(self.qx2**0.5) + "_" + str(self.qy2**0.5) + "_" + str(self.qz2**0.5)

    def __mul__(self, other):
        return Quat(self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
                    self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                    self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                    self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w)

    @staticmethod
    def identity():
        return Quat(1,0,0,0)


