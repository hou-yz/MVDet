import numpy as np


def get_worldcoord_from_imagecoord(image_coord, intrinsic_mat, extrinsic_mat):
    project_mat = intrinsic_mat @ extrinsic_mat
    project_mat = np.linalg.inv(np.delete(project_mat, 2, 1))
    image_coord = np.append(image_coord, [1])
    world_coord = (project_mat @ image_coord[:, np.newaxis]).squeeze()
    world_coord = world_coord[:2] / world_coord[2]
    return world_coord


def get_worldgrid_from_imagecoord(image_coord, intrinsic_mat, extrinsic_mat):
    worldcoord2imgcoord_mat = intrinsic_mat @ np.delete(extrinsic_mat, 2, 1)
    worldgrid2worldcoord_mat = np.array([[2.5, 0, -300], [0, 2.5, -900], [0, 0, 1]])
    worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
    imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
    image_coord = np.concatenate([image_coord, np.ones([1, image_coord.shape[1]])], axis=0)
    world_grid = (imgcoord2worldgrid_mat @ image_coord).squeeze()
    world_grid = world_grid[:2] / world_grid[2]
    return world_grid.round()


def get_imagecoord_from_worldgrid(worldgrid, intrinsic_mat, extrinsic_mat):
    worldcoord2imgcoord_mat = np.delete(intrinsic_mat @ extrinsic_mat, 2, 1)
    worldgrid2worldcoord_mat = np.array([[2.5, 0, -300], [0, 2.5, -900], [0, 0, 1]])
    worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
    worldgrid = np.append(worldgrid, [1])
    imagecoord = (worldgrid2imgcoord_mat @ worldgrid[:, np.newaxis]).squeeze()
    imagecoord = imagecoord[:2] / imagecoord[2]
    return imagecoord


def get_imagecoord_from_worldcoord(world_coord, intrinsic_mat, extrinsic_mat):
    project_mat = intrinsic_mat @ extrinsic_mat
    project_mat = np.delete(project_mat, 2, 1)
    world_coord = np.append(world_coord, [1])
    image_coord = (project_mat @ world_coord[:, np.newaxis]).squeeze()
    image_coord = image_coord[:2] / image_coord[2]
    return image_coord


def get_worldgrid_from_posid(posid):
    grid_x = posid % 480
    grid_y = posid // 480
    return np.array([grid_x, grid_y], dtype=int)


def get_posid_from_worldgrid(worldgrid):
    grid_x, grid_y = worldgrid
    return grid_x + grid_y * 480


def get_worldgrid_from_worldcoord(world_coord):
    # dataset default unit: centimeter & origin: (-300,-900)
    coord_x, coord_y = world_coord
    grid_x = (coord_x + 300) / 2.5
    grid_y = (coord_y + 900) / 2.5
    return np.array([grid_x, grid_y]).round()


def get_worldcoord_from_worldgrid(worldgrid):
    # dataset default unit: centimeter & origin: (-300,-900)
    grid_x, grid_y = worldgrid
    coord_x = -300 + 2.5 * grid_x
    coord_y = -900 + 2.5 * grid_y
    return np.array([coord_x, coord_y]).round()


def get_worldcoord_from_posid(posid):
    grid = get_worldgrid_from_posid(posid)
    return get_worldcoord_from_worldgrid(grid)


def get_posid_from_worldcoord(world_coord):
    grid = get_worldgrid_from_worldcoord(world_coord)
    return get_posid_from_worldgrid(grid)
