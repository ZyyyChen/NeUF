import numpy as np
from dataset import Quat
from utils import get_base_points,get_oriented_points_and_views
from tqdm import tqdm

import matplotlib.pyplot as plt

def get_data_from_dat_file(folder, reverse_quat):
    points = []
    viewdirs = []
    positions = []
    rotations = []
    with open(folder + "/infos.dat", "r") as info:
        lines = info.readlines()
        nb_line = 0
        for i in tqdm(range(len(lines)),total=len(lines)):
            l = lines[i]
            if l and l[0] == "D":
                dimensions = l[2:-1].split(" ")
                point_min = (float(dimensions[0]), float(dimensions[1]), float(dimensions[2]))
                point_max = (float(dimensions[3]), float(dimensions[4]), float(dimensions[5]))
                nb_line += 1
            elif l and l[0] == "m":
                splits, dimensions = l[2:-1].split(" "), []
                for element in splits:
                    try :
                        dimensions.append(float("".join([ele for ele in element if ele.isdigit() or ele == "." or ele == "-"])))
                    except ValueError :
                        pass
                point_min = (float(dimensions[0]), float(dimensions[1]), float(dimensions[2]))
                point_max = (float(dimensions[3]), float(dimensions[4]), float(dimensions[5]))
            else :

                inf = l.split('\n')[0].split(" ")

                pos = np.array(list(map(float, inf[:3])))

                if reverse_quat :
                    quat = Quat(float(inf[6]), float(inf[3]), float(inf[4]), float(inf[5]))
                else :
                    quat = Quat(float(inf[3]), -float(inf[4]), -float(inf[5]), -float(inf[6]))

                width = float(inf[7])
                height = float(inf[8])

                px_width = 192
                px_height = 512

                X, Y = get_base_points(width,height,px_width, px_height)

                p,v = get_oriented_points_and_views(X,Y, pos, quat)

                points.append(p)
                viewdirs.append(v)

                positions.append(pos)
                rotations.append(quat)

                nb_line += 1

    return np.array(positions), np.array(rotations), np.array([point_min,point_max]), np.array(points), np.array(viewdirs)

def plot_3D(positions1, positions2, bbox1, bbox2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions1[:, 0], positions1[:, 1], positions1[:, 2])
    ax.scatter(bbox1[:, 0], bbox1[:, 1], bbox1[:, 2])
    # ax.scatter(positions2[:, 0], positions2[:, 1], positions2[:, 2])
    # ax.scatter(bbox2[:, 0], bbox2[:, 1], bbox2[:, 2])
    # ax.legend(['Works','bbox Working','New', 'bbox New'])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == "__main__":

    folder_works = "c:/Users/anchling/Documents/projects/neural-ultrasound-field/datasets/bluePrintFabien_28_05_24_with_old_export"
    folder_not_working = "c:/Users/anchling/Documents/projects/Applications/out/install/x64-Release/x64/bin/data/records/test_export/bluePrintFabien_28_05_24/sync/export"

    # Load data from files if available, otherwise compute and save
    try:
        positions_good = np.load('npyfiles/positions_good.npy')
        rotations_good = np.load('npyfiles/rotations_good.npy',allow_pickle=True)
        bbox_good = np.load('npyfiles/bbox_good.npy')
        points_good = np.load('npyfiles/points_good.npy')
        viewdirs_good = np.load('npyfiles/viewdirs_good.npy')

        # positions_good_inv = np.load('npyfiles/positions_good_inv_quat.npy')
        # rotations_good_inv = np.load('npyfiles/rotations_good_inv_quat.npy',allow_pickle=True)
        # bbox_good_inv = np.load('npyfiles/bbox_good_inv_quat.npy')
        # points_good_inv = np.load('npyfiles/points_good_inv_quat.npy')
        # viewdirs_good_inv = np.load('npyfiles/viewdirs_good_inv_quat.npy')

        # positions_good_minus = np.load('npyfiles/positions_good_minus.npy')
        # rotations_good_minus = np.load('npyfiles/rotations_good_minus.npy', allow_pickle=True)
        # bbox_good_minus = np.load('npyfiles/bbox_good_minus.npy')
        # points_good_minus = np.load('npyfiles/points_good_minus.npy')
        # viewdirs_good_minus = np.load('npyfiles/viewdirs_good_minus.npy')

        positions_new = np.load('npyfiles/positions_new.npy')
        rotations_new = np.load('npyfiles/rotations_new.npy', allow_pickle=True)
        bbox_new = np.load('npyfiles/bbox_new.npy')
        points_new = np.load('npyfiles/points_new.npy')
        viewdirs_new = np.load('npyfiles/viewdirs_new.npy')

        # positions_new_inv = np.load('npyfiles/positions_new_inv_quat.npy')
        # rotations_new_inv = np.load('npyfiles/rotations_new_inv_quat.npy',allow_pickle=True)
        # bbox_new_inv = np.load('npyfiles/bbox_new_inv_quat.npy')
        # points_new_inv = np.load('npyfiles/points_new_inv_quat.npy')
        # viewdirs_new_inv = np.load('npyfiles/viewdirs_new_inv_quat.npy')

        # positions_new_minus = np.load('npyfiles/positions_new_minus.npy')
        # rotations_new_minus = np.load('npyfiles/rotations_new_minus.npy', allow_pickle=True)
        # bbox_new_minus = np.load('npyfiles/bbox_new_minus.npy')
        # points_new_minus = np.load('npyfiles/points_new_minus.npy')
        # viewdirs_new_minus = np.load('npyfiles/viewdirs_new_minus.npy')

    except FileNotFoundError:
        positions_good, rotations_good, bbox_good, points_good, viewdirs_good = get_data_from_dat_file(folder_works, False)
        positions_new, rotations_new, bbox_new, points_new, viewdirs_new = get_data_from_dat_file(folder_not_working, False)

        # Save to file the numpy arrays
        np.save('npyfiles/positions_good_minus.npy', positions_good)
        np.save('npyfiles/rotations_good_minus.npy', rotations_good)
        np.save('npyfiles/bbox_good_minus.npy', bbox_good)
        np.save('npyfiles/points_good_minus.npy', points_good)
        np.save('npyfiles/viewdirs_good_minus.npy', viewdirs_good)

        np.save('npyfiles/positions_new_minus.npy', positions_new)
        np.save('npyfiles/rotations_new_minus.npy', rotations_new)
        np.save('npyfiles/bbox_new_minus.npy', bbox_new)
        np.save('npyfiles/points_new_minus.npy', points_new)
        np.save('npyfiles/viewdirs_new_minus.npy', viewdirs_new)

    bbox_new = np.array([[106.95396182, -38.37919574, -20.36041877],[197.36367485,  19.67253138,  22.38267186]])
    plot_3D(positions_good, positions_new, bbox_good, bbox_new)

    # point_min = np.min(points_good, axis=0)
    # point_max = np.max(points_good, axis=0)

    # bounding_box = (point_min, point_max)
    # print(bounding_box)



