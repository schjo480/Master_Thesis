# Format of the hdf5 Files that store the pneuma data

This file gives a schmeatic overview of the information in `/ceph/hdd/students/yaro/new_format/`

```
/
└─── graph, Attributes=crs:str
│       └─── node_coordinates, shape=N,2
│       └─── edges, shape=M,2
│       └─── edge_features (all attributes are of shape M) (WIP, might be missing)
│       │       └─── osmid: str
│       │       └─── oneway: bool
│       │       └─── name: str
│       │       └─── highway: str
│       │       └─── reversed: bool
│       │       └─── length: float
│       │       └─── width: float
│       │       └─── lanes: str
│       │       └─── ref: str
│       │       └─── maxspeed: float
│       │       └─── junction: str
│       │       └─── access: str
│       │       └─── tunnel: str
│       └─── node_features (all attributes are of shape N) (WIP, might be missing)
│       │       └─── osmid_original: str
│       │       └─── lon: float
│       │       └─── lat: float
│       │       └─── street_count: int
│       │       └─── highway: str
└─── trajectories
│       └─── '0', 
│       │     └─── edge_idxs, shape T
│       │     └─── edge_orientation, shape T
│       │     └─── edge_observation_ranges, shape T
│       │     └─── distance_observation_to_matched_edge, shape R
│       │     └─── coordinates, shape R
│       │     └─── timestamps, shape R
│       └─── '1'
│       └───  ...
│       └─── 'K'
```
