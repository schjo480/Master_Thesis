import sys

sys.path.append('/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/src/MA_Diffusion_base_trajectory_prediction')

from utils.data_utils import TDRIVE, GEOLIFE, load_data, calculate_bbox_and_filter, \
    plot_coordinates, plot_paths, load_new_format, find_cycles, split_cycle_in_paths, \
    plot_histograms_before_after_split, \
    get_edge_used_by_trajectories, modify_and_save_data


GEOLIFE_PATH = '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/geolife.h5'
paths, node_coordinates, edges = load_new_format(GEOLIFE_PATH)
edge_coordinates = node_coordinates[edges]



def transform_data(paths, train_size=0.7, test_size=0.2, val_size=0.1, use_edge_coordinates=False, edge_coordinates=None, use_start_coordinates=False):
    output_train_file = '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/data_MID/geolife_edge_coord_train.txt'
    output_test_file = '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/data_MID/geolife_edge_coord_test.txt'
    output_val_file = '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/data_MID/geolife_edge_coord_val.txt'

    num_paths = len(paths)
    num_train = int(num_paths * train_size)
    num_test = int(num_paths * test_size)
    num_val = int(num_paths * val_size)

    train_paths = paths[:num_train]
    test_paths = paths[num_train:num_train+num_test]
    val_paths = paths[num_train+num_test:]

    if use_edge_coordinates:
        train_paths = get_edge_coordinates(train_paths, edge_coordinates)
        test_paths = get_edge_coordinates(test_paths, edge_coordinates)
        val_paths = get_edge_coordinates(val_paths, edge_coordinates)

    save_data(train_paths, output_train_file, use_edge_coordinates, use_start_coordinates)
    save_data(test_paths, output_test_file, use_edge_coordinates, use_start_coordinates)
    save_data(val_paths, output_val_file, use_edge_coordinates, use_start_coordinates)

def get_edge_coordinates(paths, edge_coordinates):
    for path in paths:
        path_edge_coordinates = edge_coordinates[path['edge_idxs']]
        path['coordinates'] = path_edge_coordinates.reshape(-1, 2)
    return paths

def save_data(paths, output_file, use_edge_coordinates=False, use_start_coordinates=False):
    with open(output_file, 'w') as outfile:
        for path in paths:
            coordinates = path['coordinates']
            timestamps = path['timestamps']
            user_idx = path['user_idx']
            user_idx = int(user_idx.decode('utf-8'))
            # taxi_idx = int(taxi_idx.decode('utf-8').split('.')[0])
            
            if use_edge_coordinates:
                if use_start_coordinates:
                    for i in range(0, len(coordinates), 2):
                        coordinate_x = coordinates[i, 0]
                        coordinate_y = coordinates[i, 1]
                        if len(timestamps) > 0:
                            if i >= len(timestamps):
                                timestamp = timestamps[-1] + 10  # create dummy timestamp
                            else:
                                timestamp = timestamps[i]
                            line = f"{timestamp}\t{user_idx}\t{coordinate_x}\t{coordinate_y}\n"
                            outfile.write(line)
                        else:
                            continue
                else:
                    for i in range(len(coordinates)):
                        coordinate_x = coordinates[i, 0]
                        coordinate_y = coordinates[i, 1]
                        if len(timestamps) > 0:
                            if i >= len(timestamps):
                                timestamp = timestamp + 10 # create dummy timestamp
                            else:
                                timestamp = timestamps[i]

                            line = f"{timestamp}\t{user_idx}\t{coordinate_x}\t{coordinate_y}\n"
                            outfile.write(line)
                        else:
                            continue
            else:
                for i in range(len(timestamps)):
                    coordinate_x = coordinates[i, 0]
                    coordinate_y = coordinates[i, 1]
                    timestamp = timestamps[i]

                    line = f"{timestamp}\t{user_idx}\t{coordinate_x}\t{coordinate_y}\n"
                    outfile.write(line)

def interpolate_timestamps(timestamps, index):
    # Implement your interpolation logic here
    # You can use libraries like numpy or scipy for interpolation
    # Return the interpolated timestamp
    pass

def interpolate_coordinates(coordinates, index):
    # Implement your interpolation logic here
    # You can use libraries like numpy or scipy for interpolation
    # Return the interpolated coordinates (coordinate_x, coordinate_y)
    pass

transform_data(paths, use_edge_coordinates=True, edge_coordinates=edge_coordinates, use_start_coordinates=True)
