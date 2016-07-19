def get_files(folder, prefix, type='.h5'):
    os.chdir(folder)
    file_list = glob.glob(prefix + '*' + type)
    return file_list


def get_index(file_list):
    regex = re.compile(r".+_x(\d\d)_y(\d\d).+")
    ind_buff = [m.group(1, 2) for l in file_list for m in [regex.search(l)] if m]
    return np.asarray(ind_buff).astype('int')


def make_grid(file_list, ver_dir=0, hor_dir=0):
    ind_list = get_index(file_list)
    x_max, y_max = ind_list.max(0)
    x_min, y_min = ind_list.min(0)
    grid = np.empty((y_max, x_max), dtype=object)
    for k_file in range(len(file_list)):
        grid[ind_list[k_file, 1] - 1, ind_list[k_file, 0] - 1] = file_list[k_file]
    if ver_dir:
        grid = np.flipud(grid)
    if hor_dir:
        grid = np.fliplr(grid)
    return grid


def start_shift_grid(file_grid, x_shift, y_shift):
    size_x = file_grid.shape[1]
    size_y = file_grid.shape[0]
    shift_grid = np.zeros([size_y, size_x, 2])
    for x_pos in range(size_x):
        shift_grid[:, x_pos, 1] = x_pos * x_shift
    ##
    for y_pos in range(size_y):
        shift_grid[y_pos, :, 0] = y_pos * y_shift
    ##
    return shift_grid


def find_pairs(file_grid):
    size_y, size_x = file_grid.shape
    pairs = np.empty((size_y * size_x, 4), dtype=object)
    for (y, x), value in np.ndenumerate(file_grid):
        nb1 = (y, x + 1)
        nb2 = (y + 1, x)
        nb3 = (y + 1, x + 1)
        if (x == size_x - 1):
            nb1 = None
            nb3 = None
        if (y == size_y - 1):
            nb2 = None
        pairs[size_x * y + x, :] = (y, x), nb1, nb2, nb3
    return pairs


g_shapes = lambda fname: h5py.File(fname, "r")['exchange/data'].shape


def get_shape(fname):
    f = h5py.File(fname, "r")
    return f['exchange/data'].shape


def refine_shift_grid(grid, shift_grid, step=100):
    if (grid.shape[0] != shift_grid.shape[0] or
                grid.shape[1] != shift_grid.shape[1]):
        return

    frame = 0
    pairs = find_pairs(grid)
    pairs_shift = pairs
    n_pairs = pairs.shape[0]

    for line in np.arange(n_pairs):
        print 'line ' + str(line), pairs[line, 0], pairs[line, 1], pairs[line, 2]
        main_pos = pairs[line, 0]
        main_shape = get_shape(grid[main_pos])
        right_pos = pairs[line, 1]
        right_shape = get_shape(grid[right_pos])
        bottom_pos = pairs[line, 2]
        bottom_shape = get_shape(grid[bottom_pos])
        diag_pos = pairs[line, 3]
        prj, flt, drk = dxchange.read_aps_32id(grid[main_pos], proj=(frame, frame + 1))
        if (main_pos[1] < 6):
            _, flt, _ = dxchange.read_aps_32id(grid[main_pos[0], 6], proj=(frame, frame + 1))
        prj = tomopy.normalize(prj, flt[20:, :, :], drk)
        prj[np.abs(prj) < 2e-3] = 2e-3
        prj[prj > 1] = 1
        prj = -np.log(prj)
        prj[np.where(np.isnan(prj) == True)] = 0
        main_prj = vig_image(prj)

        if (right_pos != None):
    prj, flt, drk = dxchange.read_aps_32id(grid[right_pos], proj=(frame, frame + 1))
            if (right_pos[0] < 6):
                _, flt, _ = dxchange.read_aps_32id(grid[right_pos[0], 6], proj=(frame, frame + 1))
    prj = tomopy.normalize(prj, flt[20:, :, :], drk)
    prj[np.abs(prj) < 2e-3] = 2e-3
            prj[prj > 1] = 1
            prj = -np.log(prj)
    prj[np.where(np.isnan(prj) == True)] = 0
    right_prj = vig_image(prj)
    shift_ini = shift_grid[right_pos] - shift_grid[main_pos]
    rangeX = shift_ini[1] + [-10, 10]
    rangeY = shift_ini[0] + [0, 5]
    right_vec = create_stitch_shift(main_prj, right_prj, rangeX, rangeY)
    pairs_shift[line, 1] = right_vec


if (bottom_pos != None):
    prj, flt, drk = dxchange.read_aps_32id(grid[bottom_pos], proj=(frame, frame + 1))
            if (bottom_pos[0] < 6):
                _, flt, _ = dxchange.read_aps_32id(grid[bottom_pos[0], 6], proj=(frame, frame + 1))
    prj = tomopy.normalize(prj, flt[20:, :, :], drk)
    prj[np.abs(prj) < 2e-3] = 2e-3
            prj[prj > 1] = 1
            prj = -np.log(prj)
    prj[np.where(np.isnan(prj) == True)] = 0
    bottom_prj = vig_image(prj)
    shift_ini = shift_grid[bottom_pos] - shift_grid[main_pos]
    rangeX = shift_ini[1] + [0, 10]
    rangeY = shift_ini[0] + [-5, 5]
    right_vec = create_stitch_shift(main_prj, bottom_prj, rangeX, rangeY)
    pairs_shift[line, 2] = right_vec
    return pairs_shift

def create_stitch_shift(block1, block2, rangeX=None, rangeY=None):
    shift_vec = np.zeros([block1.shape[0], 2])
    for frame in range(block1.shape[0]):
        shift_vec[frame, :] = cross_correlation_pcm(block1[frame, :, :], block2[frame, :, :], rangeX=rangeX,
                                                    rangeY=rangeY)
    print shift_vec
    shift = np.mean(shift_vec, 0)
    print shift
