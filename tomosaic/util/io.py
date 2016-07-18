import dxchange
import numpy as np
import tomopy


def save_partial_frame(file_grid, save_folder, prefix, frame=0):
    for (y, x), value in np.ndenumerate(file_grid):
        if (value != None):
            prj, flt, drk = dxchange.read_aps_32id(value, proj=(frame, frame + 1))
            if (x < 6):
                _, flt, _ = dxchange.read_aps_32id(file_grid[y, 6], proj=(frame, frame + 1))
            prj = tomopy.normalize(prj, flt, drk)
            fname = prefix + 'Y' + str(y).zfill(2) + '_X' + str(x).zfill(2)
            dxchange.write_tiff(np.squeeze(prj), fname=os.path.join(save_folder, fname))


def build_panorama(file_grid, shift_grid, frame=0, cam_size=[2048, 2448]):
    img_size = shift_grid[-1, -1] + cam_size
    buff = np.zeros(img_size, dtype='float16')
    for (y, x), value in np.ndenumerate(file_grid):
        if (value != None and frame < g_shapes(value)[0]):
            prj, flt, drk = dxchange.read_aps_32id(value, proj=(frame, frame + 1))
            if (x < 6):
                _, flt, _ = dxchange.read_aps_32id(file_grid[y, 6], proj=(frame, frame + 1))
            prj = tomopy.normalize(prj, flt[20:, :, :], drk)
            prj[np.abs(prj) < 2e-3] = 2e-3
            prj[prj > 1] = 1
            prj = -np.log(prj).astype('float16')
            prj[np.where(np.isnan(prj) == True)] = 0
            buff = blend(buff, np.squeeze(prj), shift_grid[y, x, :])
    return buff
