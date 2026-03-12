import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json

import numpy as np
import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import crop
from tqdm import tqdm

from utils import get_base_points, get_oriented_points_and_views

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


class Dataset:
    def __init__(self, folder, nb_valid=4, seed=-1, **kwargs):
        self.width = 0
        self.height = 0
        self.px_width = 0
        self.px_height = 0
        self.orig_px_width = 0
        self.orig_px_height = 0
        self.orig_px_size_width_mm = 0
        self.orig_px_size_height_mm = 0
        self.roi_px_size_width_mm = 0
        self.roi_px_size_height_mm = 0
        self.point_min = (0, 0, 0)
        self.point_max = (100, 100, 100)
        self.slices = []
        self.slices_valid = []
        self.X, self.Y = ([], [])
        self.name = kwargs.get("name", os.path.basename(folder))
        self.has_gt = False
        self.roi_2d = None
        self.roi_offset_x_mm = 0
        self.roi_offset_y_mm = 0
        self.front_plane_point = None
        self.front_plane_normal = None
        self.back_plane_point = None
        self.back_plane_normal = None
        self.scan_axis = None
        self.scan_length_mm = 0
        self.image_step = kwargs.get("image_step", 1)

        img_folder = kwargs.get("img_folder", "us")
        info_folder = kwargs.get("info_folder", "")
        prefix = kwargs.get("prefix", "us/img_")
        suffix = kwargs.get("suffix", ".jpg")
        reverse_quat = kwargs.get("reverse_quat", False)
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
        if seed != -1:
            np.random.seed(seed)
            torch.random.manual_seed(seed)

        first_image_path = os.path.join(folder, img_folder, prefix + "0" + suffix)
        first_image_raw = read_image(first_image_path, ImageReadMode.GRAY)
        self.orig_px_height, self.orig_px_width = first_image_raw.shape[1], first_image_raw.shape[2]
        print(f"Original image size: {self.orig_px_width} x {self.orig_px_height} px")

        infos_path = os.path.join(folder, info_folder, "infos.json")
        with open(infos_path, "r") as f:
            infos_json = json.load(f)

        infos = infos_json["infos"]
        scan_dims = infos.get("scan_dims_mm", {})
        orig_width_mm = float(scan_dims.get("width", 0.0))
        orig_height_mm = float(scan_dims.get("depth", 0.0))
        if orig_width_mm > 0 and orig_height_mm > 0:
            print(f"Original physical size: {orig_width_mm:.2f} x {orig_height_mm:.2f} mm")
            self.orig_px_size_width_mm = orig_width_mm / self.orig_px_width
            self.orig_px_size_height_mm = orig_height_mm / self.orig_px_height
            print(
                "Original pixel size: "
                f"{self.orig_px_size_width_mm:.4f} x {self.orig_px_size_height_mm:.4f} mm/px"
            )

        roi = infos.get("ROI")
        if roi and roi.get("width", 0) > 1 and roi.get("height", 0) > 1:
            self.roi_2d = roi
            print(
                f"ROI config: x={self.roi_2d['x']}, y={self.roi_2d['y']}, "
                f"w={self.roi_2d['width']}, h={self.roi_2d['height']}"
            )

        if self.roi_2d and orig_width_mm > 0 and orig_height_mm > 0:
            self.width = self.roi_2d["width"] * self.orig_px_size_width_mm
            self.height = self.roi_2d["height"] * self.orig_px_size_height_mm
            self.roi_offset_x_mm = self.roi_2d["x"] * self.orig_px_size_width_mm
            self.roi_offset_y_mm = self.roi_2d["y"] * self.orig_px_size_height_mm
            self.roi_px_size_width_mm = self.orig_px_size_width_mm
            self.roi_px_size_height_mm = self.orig_px_size_height_mm

            print("ROI cropping applied:")
            print(f"  Cropped physical size: {self.width:.2f} x {self.height:.2f} mm")
            print(f"  Physical offset: x={self.roi_offset_x_mm:.2f}mm, y={self.roi_offset_y_mm:.2f}mm")
        else:
            self.width = orig_width_mm
            self.height = orig_height_mm
            self.roi_px_size_width_mm = self.orig_px_size_width_mm
            self.roi_px_size_height_mm = self.orig_px_size_height_mm
            print("ROI cropping not applied")

        frame_keys = sorted((k for k in infos_json.keys() if k != "infos"), key=lambda k: int(k))
        selected_frame_keys = frame_keys[:: self.image_step]

        nb_line = 0
        for frame_key in tqdm(selected_frame_keys, desc="Opening dataset", total=len(selected_frame_keys)):
            frame = infos_json[frame_key]
            pos = np.array(
                [float(frame["x"]), float(frame["y"]), float(frame["z"])],
                dtype=np.float32,
            )

            if reverse_quat:
                quat = Quat(
                    float(frame["w3"]),
                    float(frame["w0"]),
                    float(frame["w1"]),
                    float(frame["w2"]),
                )
            else:
                quat = Quat(
                    float(frame["w0"]),
                    -float(frame["w1"]),
                    -float(frame["w2"]),
                    -float(frame["w3"]),
                )

            img_name = os.path.join(folder, img_folder, prefix + str(frame_key) + suffix)
            gt_name = " "
            image = self.get_torch_image(img_name)
            if os.path.exists(gt_name):
                self.has_gt = True
                gt = self.get_torch_image(gt_name)
            elif not self.has_gt:
                gt = None
            else:
                print("all or none ground truth must be provided")
                exit(-1)

            if (
                (self.px_width and self.px_width != image.shape[1])
                or (self.px_height and self.px_height != image.shape[0])
            ):
                print("images must be of consistent dimensions mm and px")
                exit(-1)

            self.px_width = image.shape[1]
            self.px_height = image.shape[0]

            self.X, self.Y = ([], [])
            if self.X == [] and self.Y == []:
                self.X, self.Y = get_base_points(
                    self.width,
                    self.height,
                    self.px_width,
                    self.px_height,
                    offset_x_mm=self.roi_offset_x_mm,
                    offset_y_mm=self.roi_offset_y_mm,
                )

            image_buffer.append(torch.squeeze(torch.reshape(image, (1, -1))))
            if self.has_gt:
                gt_buffer.append(torch.squeeze(torch.reshape(gt, (1, -1))))

            p, v = get_oriented_points_and_views(self.X, self.Y, pos, quat)

            points_buffer.append(torch.from_numpy(p.astype(dtype=np.float32)).to(device))
            viewdirs_buffer.append(torch.from_numpy(v.astype(dtype=np.float32)).to(device))

            points_numpy.append(p)
            pos_buffer.append(pos)
            rot_buffer.append(quat)
            nb_line += 1

        points_numpy = np.array(points_numpy)
        self.point_min = np.min([np.min(p, axis=0) for p in points_numpy], axis=0)
        self.point_max = np.max([np.max(p, axis=0) for p in points_numpy], axis=0)
        point_extent = self.point_max - self.point_min
        if len(pos_buffer) >= 2:
            self.front_plane_point = np.array(pos_buffer[0], dtype=np.float32)
            self.back_plane_point = np.array(pos_buffer[-1], dtype=np.float32)
            scan_vec = self.back_plane_point - self.front_plane_point
            self.scan_length_mm = float(np.linalg.norm(scan_vec))
            if self.scan_length_mm > 0:
                self.scan_axis = scan_vec / self.scan_length_mm
            else:
                self.scan_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.front_plane_normal = self.scan_axis.copy()
            self.back_plane_normal = self.scan_axis.copy()

        self.point_min_dev = torch.FloatTensor(self.point_min).to(device)
        self.point_max_dev = torch.FloatTensor(self.point_max).to(device)
        self.nb_valid = nb_valid

        i_valid = np.random.choice(range(nb_line), nb_valid, replace=False)

        j = 0
        k = 0
        nb_images = len(image_buffer)
        for i in range(nb_images - 1, 0, -1):
            if i in i_valid:
                self.slices_valid.append(
                    Slice(
                        j * self.px_width * self.px_height,
                        (j + 1) * self.px_width * self.px_height,
                        pos_buffer[i],
                        rot_buffer[i],
                    )
                )

                if self.exclude_valid:
                    image_valid_buffer.append(image_buffer.pop(i))
                    if self.has_gt:
                        gt_valid_buffer.append(gt_buffer.pop(i))
                    points_valid_buffer.append(points_buffer.pop(i))
                    viewdirs_valid_buffer.append(viewdirs_buffer.pop(i))
                    j += 1
                else:
                    self.slices.append(
                        Slice(
                            k * self.px_width * self.px_height,
                            (k + 1) * self.px_width * self.px_height,
                            pos_buffer[i],
                            rot_buffer[i],
                        )
                    )
                    k += 1

                    image_valid_buffer.append(image_buffer[i])
                    if self.has_gt:
                        gt_valid_buffer.append(gt_buffer[i])
                    points_valid_buffer.append(points_buffer[i])
                    viewdirs_valid_buffer.append(viewdirs_buffer[i])
                    j += 1
            else:
                self.slices.append(
                    Slice(
                        k * self.px_width * self.px_height,
                        (k + 1) * self.px_width * self.px_height,
                        pos_buffer[i],
                        rot_buffer[i],
                    )
                )
                k += 1

        self.pixels = torch.flatten(torch.stack(image_buffer))
        self.pixels_valid = torch.flatten(torch.stack(image_valid_buffer))
        if self.has_gt:
            self.gt = torch.flatten(torch.stack(gt_buffer))
            self.gt_valid = torch.flatten(torch.stack(gt_valid_buffer))

        self.points = torch.cat(points_buffer)
        self.points_valid = torch.cat(points_valid_buffer)
        self.viewdirs = torch.cat(viewdirs_buffer)
        self.viewdirs_valid = torch.cat(viewdirs_valid_buffer)

        self.roi_px_size_width_mm = self.width / self.px_width
        self.roi_px_size_height_mm = self.height / self.px_height

        print("\n=== Dataset loading summary ===")
        print(f"Dataset path: {folder}")
        print(f"Image count: {nb_images}")
        print(f"Final image size: {self.px_width} x {self.px_height} px")
        print(f"Final physical size: {self.width:.2f} x {self.height:.2f} mm")
        print(
            "ROI pixel size: "
            f"{self.roi_px_size_width_mm:.4f} x {self.roi_px_size_height_mm:.4f} mm/px"
        )
        print("Point range:")
        print(f"  X: {self.point_min[0]:.2f} ~ {self.point_max[0]:.2f} mm")
        print(f"  Y: {self.point_min[1]:.2f} ~ {self.point_max[1]:.2f} mm")
        print(f"  Z: {self.point_min[2]:.2f} ~ {self.point_max[2]:.2f} mm")
        print("Bounding box size:")
        print(f"  X: {point_extent[0]:.2f} mm")
        print(f"  Y: {point_extent[1]:.2f} mm")
        print(f"  Z: {point_extent[2]:.2f} mm")
        if self.front_plane_point is not None and self.back_plane_point is not None:
            print("Scan clipping planes:")
            print(
                f"  Front plane point: ({self.front_plane_point[0]:.2f}, "
                f"{self.front_plane_point[1]:.2f}, {self.front_plane_point[2]:.2f}) mm"
            )
            print(
                f"  Back plane point: ({self.back_plane_point[0]:.2f}, "
                f"{self.back_plane_point[1]:.2f}, {self.back_plane_point[2]:.2f}) mm"
            )
            print(
                f"  Plane normal / scan axis: ({self.scan_axis[0]:.6f}, "
                f"{self.scan_axis[1]:.6f}, {self.scan_axis[2]:.6f})"
            )
            print(f"  Scan length: {self.scan_length_mm:.2f} mm")

    def get_torch_image(self, img_path: str) -> torch.Tensor:
        image = read_image(img_path, ImageReadMode.GRAY)

        if self.roi_2d:
            required_keys = {"x", "y", "width", "height"}
            if not required_keys.issubset(self.roi_2d.keys()):
                raise ValueError(f"ROI must contain keys {required_keys}")
            image = crop(
                image,
                self.roi_2d["y"],
                self.roi_2d["x"],
                self.roi_2d["height"],
                self.roi_2d["width"],
            )

        return torch.squeeze(image.float()).to(device)

    def get_bounding_box(self):
        ret = (self.point_min_dev, self.point_max_dev)
        return ret

    def save(self, file_name):
        dic = {"dataset": self}
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        torch.save(dic, file_name)

    def get_slice_pixels(self, number):
        return torch.unsqueeze(self.pixels[self.slices[number].start:self.slices[number].end], 1)

    def get_slice_valid_pixels(self, number):
        return torch.unsqueeze(self.pixels_valid[self.slices_valid[number].start:self.slices_valid[number].end], 1)

    def get_slice_gt(self, number):
        if self.has_gt:
            return torch.unsqueeze(self.gt[self.slices[number].start:self.slices[number].end], 1)
        return None

    def get_slice_valid_gt(self, number):
        if self.has_gt:
            return torch.unsqueeze(self.gt_valid[self.slices_valid[number].start:self.slices_valid[number].end], 1)
        return None

    def get_slice_points(self, number):
        return torch.unsqueeze(self.points[self.slices[number].start:self.slices[number].end], 1)

    def get_slice_valid_points(self, number):
        return torch.unsqueeze(self.points_valid[self.slices_valid[number].start:self.slices_valid[number].end], 1)

    def get_slice_viewdirs(self, number):
        return torch.unsqueeze(self.viewdirs[self.slices[number].start:self.slices[number].end], 1)

    def get_slice_valid_viewdirs(self, number):
        return torch.unsqueeze(self.viewdirs_valid[self.slices_valid[number].start:self.slices_valid[number].end], 1)

    def get_indices_pixels(self, indexes):
        return torch.unsqueeze(self.pixels[indexes], 1)

    def get_indices_pixels_valid(self, indexes):
        return torch.unsqueeze(self.pixels_valid[indexes], 1)

    def get_indices_points(self, indexes):
        return torch.unsqueeze(self.points[indexes], 1)

    def get_indices_points_values(self, indexes):
        return torch.unsqueeze(self.points_valid[indexes], 1)

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

    def normalize(self):
        norm = np.sqrt(self.qw2 + self.qx2 + self.qy2 + self.qz2)
        self.w /= norm
        self.x /= norm
        self.y /= norm
        self.z /= norm
        self.compute_quat_params()

    def compute_quat_params(self):
        self.qw2 = self.w ** 2
        self.qx2 = self.x ** 2
        self.qy2 = self.y ** 2
        self.qz2 = self.z ** 2

        self.dqxqy = self.x * self.y * 2
        self.dqwqz = self.w * self.z * 2
        self.dqxqz = self.x * self.z * 2
        self.dqwqy = self.w * self.y * 2
        self.dqyqz = self.y * self.z * 2
        self.dqwqx = self.w * self.x * 2

    def apply_quat(self, point):
        return np.array(
            [
                point[0] * (self.qw2 + self.qx2 - self.qy2 - self.qz2)
                + point[1] * (self.dqxqy + self.dqwqz)
                + point[2] * (self.dqxqz - self.dqwqy),
                point[0] * (self.dqxqy - self.dqwqz)
                + point[1] * (self.qw2 - self.qx2 + self.qy2 - self.qz2)
                + point[2] * (self.dqyqz + self.dqwqx),
                point[0] * (self.dqxqz + self.dqwqy)
                + point[1] * (self.dqyqz - self.dqwqx)
                + point[2] * (self.qw2 - self.qx2 - self.qy2 + self.qz2),
            ]
        )

    def as_rotmat(self):
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array(
            [
                [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)],
            ]
        )

    def __repr__(self):
        return str(self.qw2 ** 0.5) + "_" + str(self.qx2 ** 0.5) + "_" + str(self.qy2 ** 0.5) + "_" + str(self.qz2 ** 0.5)

    def __mul__(self, other):
        return Quat(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    @staticmethod
    def identity():
        return Quat(1, 0, 0, 0)
