import h5py
import math
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

GEOLIFE, TDRIVE = 'geolife', 'tdrive'


def load_data(merged_path, WHICH):
    paths = []

    with h5py.File(merged_path) as hf:
        node_coordinates = hf['graph']['node_coordinates'][:]
        edges = hf['graph']['edges'][:]

        if WHICH == GEOLIFE:
            for user_idx in tqdm(list(hf['trajectories'].keys())):
                for record_file in hf['trajectories'][user_idx].keys():
                    for subtrajectory_idx in hf['trajectories'][user_idx][record_file]['subtrajectories'].keys():
                        subtrajectory_group = hf['trajectories'][user_idx][record_file]['subtrajectories'][
                            subtrajectory_idx]
                        path = {attr: subtrajectory_group[attr][:] for attr in subtrajectory_group.keys()} | {
                            'user_idx': user_idx, 'record_file': record_file}
                        paths.append(path)
        elif WHICH == TDRIVE:
            # example object list(hf3['trajectories']['10028.txt']['subtrajectories']['0'].items())
            for taxi_idx in tqdm(list(hf['trajectories'].keys())):
                for subtrajectory_idx in hf['trajectories'][taxi_idx]['subtrajectories'].keys():
                    subtrajectory_group = hf['trajectories'][taxi_idx]['subtrajectories'][subtrajectory_idx]
                    path = {attr: subtrajectory_group[attr][:] for attr in subtrajectory_group.keys()} | {
                        'taxi_idx': taxi_idx}
                    paths.append(path)
        return paths, node_coordinates, edges


def plot_coordinates(all_coordinates, edges, node_coordinates):
    fig, ax = plt.subplots(figsize=(9, 9))
    for ec in node_coordinates[edges]:
        ax.plot(ec[:, 0], ec[:, 1], c='grey', lw=.5)
    ax.scatter(all_coordinates[:, 0], all_coordinates[:, 1], s=1, c='red')


def calculate_bbox_and_filter(all_coordinates, percentage_to_keep_by_axis):
    lower_percentage = (100 - percentage_to_keep_by_axis) / 2
    upper_percentage = 100 - lower_percentage

    min_x = np.percentile(all_coordinates[:, 0], lower_percentage)
    min_y = np.percentile(all_coordinates[:, 1], lower_percentage)
    max_x = np.percentile(all_coordinates[:, 0], upper_percentage)
    max_y = np.percentile(all_coordinates[:, 1], upper_percentage)

    bbox = [(min_x, min_y), (max_x, max_y)]

    mask = (all_coordinates[:, 0] >= min_x) & (all_coordinates[:, 0] <= max_x) & \
           (all_coordinates[:, 1] >= min_y) & (all_coordinates[:, 1] <= max_y)
    filtered_coordinates = all_coordinates[mask]

    return bbox, filtered_coordinates


def plot_paths(paths, node_coordinates, edges, num_paths_to_plot=4, random=False, start_id=0, zoom_in=True):
    fig, axs = plt.subplots(2, (num_paths_to_plot + 1) // 2, figsize=(10, 10),
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.1})
    edge_coordinates = node_coordinates[edges]
    start_idx = np.random.randint(0, len(paths) - num_paths_to_plot) if random else start_id

    if random:
        print(f"Starting from index {start_idx}")

    for ax in tqdm(axs.flatten()):
        path = paths[start_idx]
        start_idx += 1

        trajectory_coordinates = path['coordinates']
        trajectory_timestamps = path['timestamps']
        trajectory_edge_orientations = path['edge_orientations']
        trajectory_edge_idxs = path['edge_idxs']

        if zoom_in:
            traj_edge_coordinates = node_coordinates[edges[trajectory_edge_idxs]]
            xmin, xmax = traj_edge_coordinates.min(axis=0), traj_edge_coordinates.max(axis=0)

        else:
            xmin, xmax = node_coordinates.min(axis=0), node_coordinates.max(axis=0)

        edge_mask = (edge_coordinates >= xmin[None, :] - 0.01).all(-1).any(-1) & (
                edge_coordinates <= xmax[None, :] + 0.01).all(-1).any(-1)

        all_c = node_coordinates[edges[edge_mask]].reshape((-1, 2))
        arrow_width = (all_c.max(axis=0) - all_c.min(axis=0)).max() * 0.01

        # Plot all edges (i.e. streets)
        for ec, edge_idx in zip(edge_coordinates[edge_mask], np.arange(edges.shape[0], dtype=int)[edge_mask]):
            ax.plot(ec[:, 0], ec[:, 1], c='grey', lw=.5)  # , c=1 - np.array((c, c, c, 0)))
        # Plot trajectory
        for ec, orient in zip(edge_coordinates[trajectory_edge_idxs], trajectory_edge_orientations):
            ax.arrow(*ec.mean(axis=0), *(np.diff(ec, axis=0).flatten() * 0.1 * orient), shape='full', ec='none',
                     fc='red',
                     width=0, head_width=arrow_width, zorder=2)

        if len(trajectory_coordinates) > 0:
            c = trajectory_timestamps - trajectory_timestamps.min()
            c /= c.max() if c.max() > 0 else 1
            ax.scatter(trajectory_coordinates[:, 0], trajectory_coordinates[:, 1], alpha=.9, marker='x', s=10, c=c,
                       cmap='viridis')

        ax.set_xticks([])
        ax.set_yticks([])


def find_cycles(paths):
    cycles = []
    for path in tqdm(paths):
        edge_idxs = path['edge_idxs']
        unique_edges, counts = np.unique(edge_idxs, return_counts=True)
        cycles_in_path = unique_edges[counts > 1]
        if cycles_in_path.size > 0:
            cycles.append(True)
        else:
            cycles.append(False)
    return np.array(cycles)


def split_path(path, start, end):
    if len(path['edge_observation_ranges']) == 0:
        coordinate_split_start = 0
        coordinate_split_end = 0
    else:
        coordinate_split_start = path['edge_observation_ranges'][start][0]
        coordinate_split_end = path['edge_observation_ranges'][end - 1][1]

    return path | {"edge_idxs": path["edge_idxs"][start:end],
                   "edge_orientations": path["edge_orientations"][start:end],
                   "edge_observation_ranges": path["edge_observation_ranges"][start:end],
                   "coordinates": path["coordinates"][coordinate_split_start:coordinate_split_end],
                   "timestamps": path["timestamps"][coordinate_split_start:coordinate_split_end],
                   "distance_observation_to_matched_edge": path["distance_observation_to_matched_edge"][
                                                           coordinate_split_start:coordinate_split_end],
                   }


def split_cycle_in_paths(paths):
    split_paths = []
    for path in tqdm(paths):
        edge_idxs = path['edge_idxs']
        seen_edges = {}
        start_of_path = 0
        for i, edge in enumerate(edge_idxs):
            if edge in seen_edges:
                cycle_start = seen_edges[edge]
                # split cycle on average point in the cycle, not to lose information, alternatively
                # we could split on the first occurence of the cycle by changing
                # (cycle_start + i) // 2 to i
                split_idx = math.ceil((cycle_start + i) / 2)
                before_cycle = split_path(path, start_of_path, split_idx)
                split_paths.append(before_cycle)
                start_of_path = split_idx
                seen_edges = {edge: i for edge, i in seen_edges.items() if i > cycle_start}

            seen_edges[edge] = i
        split_paths.append(split_path(path, start_of_path, len(edge_idxs)))
    return split_paths


def load_new_format(new_file_path):
    paths = []

    with h5py.File(new_file_path, 'r') as new_hf:
        node_coordinates = new_hf['graph']['node_coordinates'][:]
        edges = new_hf['graph']['edges'][:]
        
        for i in tqdm(new_hf['trajectories'].keys()):
                path_group = new_hf['trajectories'][i]
                path = {attr: path_group[attr][()] for attr in path_group.keys()}
                if 'edge_orientation' in path:
                    path['edge_orientations'] = path.pop('edge_orientation')
                paths.append(path)

    return paths, node_coordinates, edges


def plot_histograms_before_after_split(paths_before_split, paths_after_split):
    # Calculate number of edges in each path before split
    num_edges_before_split = [len(path['edge_idxs']) for path in paths_before_split]

    # Calculate number of edges in each path after split
    num_edges_after_split = [len(path['edge_idxs']) for path in paths_after_split]

    # Plot histograms
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(num_edges_before_split, bins=75, log=True, alpha=0.7)
    plt.title('Before Split')
    plt.xlabel('Number of Edges')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(num_edges_after_split, bins=75, log=True, alpha=0.7)
    plt.title('After Split')
    plt.xlabel('Number of Edges')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def get_edge_used_by_trajectories(paths):
    edge_used_by_trajectories = set()
    for path in (paths):
        edge_used_by_trajectories.update(path['edge_idxs'])
    return np.array(sorted(list(edge_used_by_trajectories)))


def copy_and_create_datasets(old_hf, new_hf, modified_node_coordinates, modified_edges, modified_paths):
    old_hf.copy('graph', new_hf)

    del new_hf['graph/node_coordinates']
    del new_hf['graph/edges']
    new_hf.create_dataset('graph/node_coordinates', data=modified_node_coordinates)
    #new_hf.create_dataset('graph/edges', data=modified_edges)
    new_hf['graph'].create_dataset('edge_used_by_trajectory', data=get_edge_used_by_trajectories(modified_paths))


def flatten_node_features(new_hf, node_mapping=None):
    for node_type in tqdm(new_hf['graph']['node_features'].keys(), desc='Flattening node features'):
        flat_features = []
        for node in sorted(list(map(int, new_hf['graph']['node_features'][node_type].keys()))):
            if node_mapping and node not in node_mapping:
                continue
            mapped_node = node_mapping[node] if node_mapping else node
            node_feat = new_hf['graph']['node_features'][node_type][str(mapped_node)][()]
            flat_features.append(node_feat[0] if len(node_feat) == 1 else np.nan)
        del new_hf['graph']['node_features'][node_type]
        new_hf['graph']['node_features'].create_dataset(node_type, data=np.array(flat_features))


def flatten_edge_features(new_hf, edge_mapping=None):
    for road_type in tqdm(new_hf['graph']['edge_features'].keys(), desc='Flattening edge features'):
        flat_features = []
        for edge in sorted(list(map(int, new_hf['graph']['edge_features'][road_type].keys()))):
            if edge_mapping and edge not in edge_mapping:
                continue
            mapped_edge = edge_mapping[edge] if edge_mapping else edge
            edge_feat = new_hf['graph']['edge_features'][road_type][str(mapped_edge)][()]
            if road_type == 'highway':
                flat_features.append(str([st.decode('utf-8') for st in edge_feat]).encode('utf-8'))
            else:
                flat_features.append(edge_feat[0] if len(edge_feat) == 1 else np.nan)
        del new_hf['graph']['edge_features'][road_type]
        new_hf['graph']['edge_features'].create_dataset(road_type, data=np.array(flat_features))


def create_trajectories_group(new_hf, modified_paths):
    trajectories_group = new_hf.create_group('trajectories')
    for i, path in enumerate(tqdm(modified_paths, desc='Creating trajectories group')):
        path_group = trajectories_group.create_group(str(i))
        for attr, data in path.items():
            path_group.create_dataset(attr, data=data)


def modify_and_save_data(old_file_path, new_file_path, modified_paths, modified_node_coordinates, modified_edges,
                         node_mapping=None, edge_mapping=None):
    with h5py.File(old_file_path, 'r') as old_hf, h5py.File(new_file_path, 'w') as new_hf:
        copy_and_create_datasets(old_hf, new_hf, modified_node_coordinates, modified_edges, modified_paths)
        flatten_node_features(new_hf, node_mapping)
        flatten_edge_features(new_hf, edge_mapping)
        create_trajectories_group(new_hf, modified_paths)


def remove_bad_matches(paths, threshold_percentile: int = 95, mode: str = 'max'):
    max_dist_abs = []
    mean_dist = []
    for i in range(len(paths)):
        if len(paths[i]['distance_observation_to_matched_edge']) > 0:
            max_dist_abs.append(np.max(paths[i]['distance_observation_to_matched_edge']))
            mean_dist.append(np.mean(paths[i]['distance_observation_to_matched_edge']))

    new_paths = []
    bad_matches = []
    if len(max_dist_abs) > 0:
        if mode == 'max':
            threshold = np.percentile(max_dist_abs, threshold_percentile)
            for path in tqdm(paths):
                if (len(path['distance_observation_to_matched_edge']) > 0):
                    if np.max(path['distance_observation_to_matched_edge']) < threshold:
                        new_paths.append(path)
                    else:
                        bad_matches.append(path)
        elif mode == 'mean':
            threshold = np.percentile(mean_dist, threshold_percentile)
            for path in tqdm(paths):
                if (len(path['distance_observation_to_matched_edge']) > 0):
                    if np.mean(path['distance_observation_to_matched_edge']) < threshold:
                        new_paths.append(path)
                    else:
                        bad_matches.append(path)
                
    return new_paths, bad_matches
