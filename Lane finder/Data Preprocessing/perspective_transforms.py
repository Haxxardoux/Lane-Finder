import matplotlib.pyplot as plt
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.nuscenes import NuScenes
import nuscenes
from nuscenes.utils.geometry_utils import view_points

from matplotlib.patches import Rectangle, Arrow
import descartes
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from pyquaternion import Quaternion
from PIL import Image
import numpy as np

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


nusc = NuScenes(version='v1.0-mini', dataroot='C:\\Users\\turbo\\Python projects\\Lane finder\\data\\v1.0-mini', verbose=True)
nusc_map = NuScenesMap(dataroot="C:/Users/turbo/Python projects/Lane finder/data/v1.0-mini", map_name='singapore-onenorth')
map_api = NuScenesMapExplorer(nusc_map)

# get some random scene, only like 4 total, can do nusc.list_scenes() to see them
my_scene = nusc.scene[0]

# get the front camera tokens at each time step for the scene
tokens = []
for i in range(400):
    tokens.append(nusc.sample[i]['data']['CAM_FRONT'])

for token in tokens[::10]:
    List = []
    List2 = []

    # not sure tbh :/
    im = Image.open(nusc.get_sample_data_path(token))
    plt.imshow(im)
    im_size = im.size
    cam_record = nusc.get('sample_data', token)
    cs_record = nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    poserecord = nusc.get('ego_pose', cam_record['ego_pose_token'])

    # starting from the current position of the sensor in global coordinates (pose), find the nearby points (+/- 10000)
    ego_pose = poserecord['translation']
    box_coords = (
        ego_pose[0] - 10000,
        ego_pose[1] - 10000,
        ego_pose[0] + 10000,
        ego_pose[1] + 10000,
    )

    # find the stuff around the sensor that we care about, in this case a wide radius so it should give us the whole map
    records_in_patch = nusc_map.get_records_in_patch(box_coords, ['road_segment', 'lane'], 'within')
    records_in__patch = nusc_map.get_records_in_patch(box_coords, nusc_map.non_geometric_layers, mode='intersect')

    for layer_name in ['road_divider', 'lane_divider']:
        for token in records_in__patch[layer_name]:
            record = map_api.map_api.get(layer_name, token)
            line = map_api.map_api.extract_line(record['line_token'])
            
            # Convert polygon nodes to pointcloud with 0 height.
            points = np.array(line.xy)
            points = np.vstack((points, np.zeros((1, points.shape[1]))))

            List.append(points)
            
            # Transform into the ego vehicle frame for the timestamp of the image.
            points = np.vstack((points, np.ones((1,points.shape[1])) ))
            mt = transform_matrix(poserecord['translation'], Quaternion(poserecord['rotation']), inverse=True)
            points = np.dot(mt, points)

            # Transform into camera
            mt = transform_matrix(cs_record['translation'], Quaternion(cs_record['rotation']), inverse=True)
            points = np.dot(mt, points)
            points = points[:3, :]
            points = view_points(points, cam_intrinsic, normalize=True)
            
            
            # ignore lines that are behind camera 
            inside = np.ones(points.shape[1], dtype=bool)
            inside = np.logical_and(inside, points[0, :] > 1)
            inside = np.logical_and(inside, points[0, :] < im.size[0] - 1)
            inside = np.logical_and(inside, points[1, :] > 1)
            inside = np.logical_and(inside, points[1, :] < im.size[1] - 1)

            # temp
            # break
            if np.any(np.logical_not(inside)):
                continue

            List2.append(points)

    # plot the whole map
    points = np.concatenate(List2, axis=1)
    # temp = points[:2].T
    # plt.scatter(temp[:, 0], temp[:, 1])

    # # transform image with inverse intrinsic camera matrix
    # points = np.concatenate(List2, axis=1)
    # points = view_points(points, np.linalg.inv(cam_intrinsic), False)
    # points = np.vstack((points, np.ones((1,points.shape[1])) ))

    # # transform perspective from camera to sensor
    # mt = transform_matrix(cs_record['translation'], Quaternion(cs_record['rotation']))
    # points = np.dot(mt, points)

    # # transform from sensor to ego pose
    # mt = transform_matrix(poserecord['translation'], Quaternion(poserecord['rotation']))
    # points = np.dot(mt, points)


    plt.scatter(points[2]*points[0], points[1])
    plt.show()
