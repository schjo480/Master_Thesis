import numpy as np
import sys
import os
import pandas as pd
import dill
import pickle

from environment import Environment, Scene, Node, derivative_of

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    }
}

def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise

def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        data_dict = {('position', 'x'): x,
                     ('position', 'y'): y,
                     ('velocity', 'x'): vx,
                     ('velocity', 'y'): vy,
                     ('acceleration', 'x'): ax,
                     ('acceleration', 'y'): ay}

        node_data = pd.DataFrame(data_dict, columns=data_columns)

        node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


data_folder_name = 'processed_data_noise'

maybe_makedirs(data_folder_name)
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

# Process ETH-UCY
# for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
for desired_source in ['tdrive_edge_coordinates']:
    for data_class in ['train', 'val', 'test']:
        '''attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
        env.attention_radius = attention_radius'''

        scenes = []
        data_folder_name_no_social = 'processed_data_no_social'
        maybe_makedirs(data_folder_name_no_social)
        data_dict_path = os.path.join(data_folder_name_no_social, '_'.join([desired_source, data_class]) + 'custom.pkl')

        for subdir, dirs, files in os.walk(os.path.join('data/data_MID', desired_source, data_class)):
            for file in files:
                if file.endswith('.txt'):
                    input_data_dict = dict()
                    full_data_path = os.path.join(subdir, file)
                    print('At', full_data_path)

                    data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None)
                    
                    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
                    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')

                    # data['frame_id'] = data['frame_id'] // 10

                    data['frame_id'] -= data['frame_id'].min()

                    # data.sort_values('frame_id', inplace=True)

                    data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
                    data['pos_y'] = data['pos_y'] - data['pos_y'].mean()
                    
                    dx = np.diff(data['pos_x'])
                    dy = np.diff(data['pos_y'])
                    dt = np.abs(np.diff(data['frame_id']))

                    vx, vy, ax, ay = [], [], [], []
                    vx.append(0)
                    vy.append(0)
                    ax.append(0)
                    ay.append(0)
                    track_change = np.diff(data['track_id']) != 0
                    for i in range(1, len(data.index) - 1):
                        vx.append(dx[i] / dt[i])
                        ax.append(vx[i] - vx[i-1] / dt[i])
                        vy.append(dy[i] / dt[i])
                        ay.append(vy[i] - vy[i-1] / dt[i])
                        vx[track_change[i]] = 0
                        ax[track_change[i]] = 0
                        vy[track_change[i]] = 0
                        ay[track_change[i]] = 0
                    vx.append(0)
                    vy.append(0)
                    ax.append(0)
                    ay.append(0)
                    data['v_x'] = vx
                    data['a_x'] = ax
                    data['v_y'] = vy
                    data['a_y'] = ay
                    data.drop(columns=['track_id'], inplace=True)
                    
                    data_dict = {('position', 'x'): data['pos_x'].values,
                                    ('position', 'y'): data['pos_y'].values,
                                    ('velocity', 'x'): data['v_x'].values,
                                    ('velocity', 'y'): data['v_y'].values,
                                    ('acceleration', 'x'): data['a_x'].values,
                                    ('acceleration', 'y'): data['a_y'].values}

                    node_data = pd.DataFrame(data_dict, columns=data_columns)
                    # print(node_data.head(50).to_string())

        print(f'Processed {len(data):.2f} recordings for dataset {desired_source}')

        with open(data_dict_path, 'wb') as f:
            dill.dump(node_data, f, protocol=dill.HIGHEST_PROTOCOL)
exit()
# Process Stanford Drone. Data obtained from Y-Net github repo
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])


for data_class in ["train", "test"]:
    raw_path = "raw_data/stanford"
    out_path = "processed_data"
    data_path = os.path.join(raw_path, f"{data_class}_trajnet.pkl")
    print(f"Processing SDD {data_class}")
    data_out_path = os.path.join(out_path, f"sdd_{data_class}.pkl")
    df = pickle.load(open(data_path, "rb"))
    env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
    env.attention_radius = attention_radius

    scenes = []

    group = df.groupby("sceneId")

    for scene, data in group:
        data['frame'] = pd.to_numeric(data['frame'], downcast='integer')
        data['trackId'] = pd.to_numeric(data['trackId'], downcast='integer')

        data['frame'] = data['frame'] // 12

        data['frame'] -= data['frame'].min()

        data['node_type'] = 'PEDESTRIAN'
        data['node_id'] = data['trackId'].astype(str)

        # apply data scale as same as PECnet
        data['x'] = data['x']/50
        data['y'] = data['y']/50

        # Mean Position
        data['x'] = data['x'] - data['x'].mean()
        data['y'] = data['y'] - data['y'].mean()

        max_timesteps = data['frame'].max()

        if len(data) > 0:

            scene = Scene(timesteps=max_timesteps+1, dt=dt, name="sdd_" + data_class, aug_func=augment if data_class == 'train' else None)
            n=0
            for node_id in pd.unique(data['node_id']):

                node_df = data[data['node_id'] == node_id]


                if len(node_df) > 1:
                    assert np.all(np.diff(node_df['frame']) == 1)
                    if not np.all(np.diff(node_df['frame']) == 1):
                        pdb.set_trace()

                    node_values = node_df[['x', 'y']].values

                    if node_values.shape[0] < 2:
                        continue

                    new_first_idx = node_df['frame'].iloc[0]

                    x = node_values[:, 0]
                    y = node_values[:, 1]
                    vx = derivative_of(x, scene.dt)
                    vy = derivative_of(y, scene.dt)
                    ax = derivative_of(vx, scene.dt)
                    ay = derivative_of(vy, scene.dt)

                    data_dict = {('position', 'x'): x,
                                 ('position', 'y'): y,
                                 ('velocity', 'x'): vx,
                                 ('velocity', 'y'): vy,
                                 ('acceleration', 'x'): ax,
                                 ('acceleration', 'y'): ay}

                    node_data = pd.DataFrame(data_dict, columns=data_columns)
                    node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
                    node.first_timestep = new_first_idx

                    scene.nodes.append(node)
            if data_class == 'train':
                scene.augmented = list()
                angles = np.arange(0, 360, 15) if data_class == 'train' else [0]
                for angle in angles:
                    scene.augmented.append(augment_scene(scene, angle))

            print(scene)
            scenes.append(scene)
    env.scenes = scenes

    if len(scenes) > 0:
        with open(data_out_path, 'wb') as f:
            #pdb.set_trace()
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
