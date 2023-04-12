from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.optimize import root
from .helpers import phi_prime


def adjust_plot(ax, xmin, xmax, ymin, ymax):
    ax.set_xlim(xmin - 0.05 * (xmax - xmin), xmax + 0.05 * (xmax - xmin))
    ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))


def plot_field(
    net,
    vec1=None,
    vec2=None,
    xmin=-3,
    xmax=3,
    ymin=-3,
    ymax=3,
    input=None,
    res=50,
    ax=None,
    add_fixed_points=False,
    fixed_points_trials=10,
    fp_save=None,
    fp_load=None,
    nojac=False,
    orth=False,
    alt_naming=True,
    sizes=1,
):
    """
    Plot the flow field of a rank 2 network in its (m1, m2) plane (eventually affine if there is an input)
    Note: assumes the net uses tanh non-linearity
    Note 2: if plotting fixed points, stability calculations only make sense in the case of a rank2 net in plane
    (m1, m2)
    :param net:
    :param vec1: numpy array, if None, the m1 vector of the network is selected
    :param vec2: numpy array, if None, the m2 vector of the net is selected
    :param xmin: float
    :param xmax: float
    :param ymin: float
    :param ymax: float
    :param input: tensor of shape (dim_input) that is used to compute input vector
    :param res: int, resolution of flow field
    :param ax: matplotlib.Axes, plot on it if given
    :param add_fixed_points: bool
    :param fixed_points_trials: int, number of trials for fixed points search
    :param nojac: if True, do not use the jacobian for the fixed points finder
    :param orth: bool, True to orthogonalize vec2 with vec1
    :param alt_naming: set True for SupportLowRankRNN_withMask
    :return: matplotlib.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    adjust_plot(ax, xmin, xmax, ymin, ymax)
    if vec1 is None:
        vec1 = net.m[:, 0].squeeze().detach().numpy()
    if vec2 is None:
        vec2 = net.m[:, 1].squeeze().detach().numpy()
    if add_fixed_points:
        n1 = net.n[:, 0].squeeze().detach().numpy()
        n2 = net.n[:, 1].squeeze().detach().numpy()
    if hasattr(net, "wrec") and net.wrec is not None:
        w_rec = net.wrec.detach().numpy()
    else:
        w_rec = None
        m = net.m.detach().numpy()
        n = net.n.detach().numpy().T
        if alt_naming:
            m = net.m_rec.detach().numpy()
            n = net.n_rec.detach().numpy().T

    # Plotting constants
    marker_size = 50 * sizes
    nx, ny = res, res

    # Orthogonalization of the basis vec1, vec2, I
    if orth:
        vec2 = vec2 - (vec2 @ vec1) * vec1 / (vec1 @ vec1)
    if input is not None:
        I = (input @ net.wi_full).detach().numpy()
        I_orth = I - (I @ vec1) * vec1 / (vec1 @ vec1) - (I @ vec2) * vec2 / (vec2 @ vec2)
    else:
        I = np.zeros(net.hidden_size)
        I_orth = np.zeros(net.hidden_size)

    # rescaling factors (for transformation euclidean space / overlap space)
    # here, if one wants x s.t. overlap(x, vec1) = alpha, x should be r1 * alpha * vec1
    # with the overlap being defined as overlap(u, v) = u.dot(v) / sqrt(hidden_size)
    r1 = sqrt(net.hidden_size) / (vec1 @ vec1)
    r2 = sqrt(net.hidden_size) / (vec2 @ vec2)

    # Defining the grid
    xs_grid = np.linspace(xmin, xmax, nx + 1)
    ys_grid = np.linspace(ymin, ymax, ny + 1)
    xs = (xs_grid[1:] + xs_grid[:-1]) / 2
    ys = (ys_grid[1:] + ys_grid[:-1]) / 2
    field = np.zeros((nx, ny, 2))
    X, Y = np.meshgrid(xs, ys)

    # Recurrent function of dx/dt = F(x, I)
    if w_rec is not None:

        def F(x, I):
            return -x + w_rec @ np.tanh(x) + I

    else:

        def F(x, I):
            return -x + m @ (n @ np.tanh(x)) + I

    # Derivative of tanh
    def phiPrime(x):
        return 1 - np.tanh(x) ** 2

    # Jacobian of F, assuming F is rank 2
    def FJac(x, I=None):
        phiPr = phiPrime(x)
        n1_eff = n1 * phiPr
        n2_eff = n2 * phiPr
        return np.outer(vec1, n1_eff) + np.outer(vec2, n2_eff) - np.identity(net.hidden_size)

    # Compute flow in each point of the grid
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            h = r1 * x * vec1 + r2 * y * vec2 + I_orth
            delta = F(h, I)
            field[j, i, 0] = delta @ vec1 / sqrt(net.hidden_size)
            field[j, i, 1] = delta @ vec2 / sqrt(net.hidden_size)

    ax.streamplot(
        xs,
        ys,
        field[:, :, 0],
        field[:, :, 1],
        color="white",
        density=0.5,
        arrowsize=sizes,
        linewidth=sizes * 0.8,
    )

    norm_field = np.sqrt(field[:, :, 0] ** 2 + field[:, :, 1] ** 2)
    mappable = ax.pcolor(X, Y, norm_field)

    # Look for fixed points
    if add_fixed_points:
        if fp_load is None:
            stable_sols = []
            saddles = []
            sources = []

            # initial conditions are dispersed over a grid
            X_grid, Y_grid = np.meshgrid(
                np.linspace(xmin, xmax, int(sqrt(fixed_points_trials))),
                np.linspace(ymin, ymax, int(sqrt(fixed_points_trials))),
            )

            for i in range(X_grid.size):
                xy = X_grid.ravel()[i], Y_grid.ravel()[i]
                x0 = r1 * xy[0] * vec1 + r2 * xy[1] * vec2 + I_orth
                sol = root(F, x0, args=I, jac=None if nojac else FJac)

                # if solution found
                if sol.success == 1:
                    kappa_sol = [
                        (sol.x @ vec1) / sqrt(net.hidden_size),
                        (sol.x @ vec2) / sqrt(net.hidden_size),
                    ]
                    # Computing stability
                    pseudoJac = np.zeros((2, 2))
                    phiPr = phiPrime(sol.x)
                    n1_eff = n1 * phiPr
                    n2_eff = n2 * phiPr
                    pseudoJac[0, 0] = vec1 @ n1_eff
                    pseudoJac[0, 1] = vec2 @ n1_eff
                    pseudoJac[1, 0] = vec1 @ n2_eff
                    pseudoJac[1, 1] = vec2 @ n2_eff
                    eigvals = np.linalg.eigvals(pseudoJac)
                    if np.all(np.real(eigvals) <= 1):
                        stable_sols.append(kappa_sol)
                    elif np.any(np.real(eigvals) <= 1):
                        saddles.append(kappa_sol)
                    else:
                        sources.append(kappa_sol)
        # Load fixed points stored in a file
        else:
            arrays = np.load(fp_load)
            arr = arrays["arr_0"]
            stable_sols = [arr[i] for i in range(arr.shape[0])]
            arr = arrays["arr_1"]
            saddles = [arr[i] for i in range(arr.shape[0])]
            arr = arrays["arr_2"]
            sources = [arr[i] for i in range(arr.shape[0])]
            print(saddles)
        if fp_save is not None:
            np.savez(fp_save, np.array(stable_sols), np.array(saddles), np.array(sources))
        else:
            ax.scatter(
                [x[0] for x in stable_sols],
                [x[1] for x in stable_sols],
                facecolors="white",
                edgecolors="white",
                s=marker_size,
                zorder=1000,
            )
            ax.scatter(
                [x[0] for x in saddles],
                [x[1] for x in saddles],
                facecolors="black",
                edgecolors="white",
                s=marker_size,
                zorder=1000,
            )
            ax.scatter(
                [x[0] for x in sources],
                [x[1] for x in sources],
                facecolors="black",
                edgecolors="white",
                s=marker_size,
                zorder=1000,
            )
    return ax, mappable


#
# def plot_field2(net, vec1=None, vec2=None, xmin=-3, xmax=3, ymin=-3, ymax=3, input=None, res=50,
#                ax=None, add_fixed_points=False, fixed_points_trials=10, nojac=False, orth=False):
#     if ax is None:
#         fig, ax = plt.subplots()
#     adjust_plot(ax, xmin, xmax, ymin, ymax)
#     if net.wrec is not None:
#         w_rec = net.wrec.detach().numpy()
#     else:
#         w_rec = None
#         m = net.m.detach().numpy()
#         n = net.n.detach().numpy().T
#
#     # Plotting constants
#     marker_size = 90
#     nx, ny = res, res
#
#     # Orthogonalization of the basis vec1, vec2, I
#     if orth:
#         vec2 = vec2 - (vec2 @ vec1) * vec1 / (vec1 @ vec1)
#     if input is not None:
#         I = (input @ net.wi_full).detach().numpy()
#         I_orth = I - (I @ vec1) * vec1 / (vec1 @ vec1) - (I @ vec2) * vec2 / (vec2 @ vec2)
#     else:
#         I = np.zeros(net.hidden_size)
#         I_orth = np.zeros(net.hidden_size)
#
#     # rescaling factors (for transformation euclidean space / overlap space)
#     # here, if one wants x s.t. overlap(x, vec1) = alpha, x should be r1 * alpha * vec1
#     # with the overlap being defined as overlap(u, v) = u.dot(v) / sqrt(hidden_size)
#     r1 = 1. / (vec1 @ vec1)
#     r2 = 1. / (vec2 @ vec2)
#
#     # Defining the grid
#     xs_grid = np.linspace(xmin, xmax, nx + 1)
#     ys_grid = np.linspace(ymin, ymax, ny + 1)
#     xs = (xs_grid[1:] + xs_grid[:-1]) / 2
#     ys = (ys_grid[1:] + ys_grid[:-1]) / 2
#     field = np.zeros((nx, ny, 2))
#     X, Y = np.meshgrid(xs, ys)
#
#     # Recurrent function of dx/dt = F(x, I)
#     if w_rec is not None:
#         def F(x, I):
#             return -x + w_rec @ np.tanh(x) + I
#     else:
#         def F(x, I):
#             return -x + m @ (n @ np.tanh(x)) + I
#
#     # Compute flow in each point of the grid
#     for i, x in enumerate(xs):
#         for j, y in enumerate(ys):
#             h = r1 * x * vec1 + r2 * y * vec2 + I_orth
#             delta = F(h, I)
#             field[j, i, 0] = delta @ vec1
#             field[j, i, 1] = delta @ vec2
#     ax.streamplot(xs, ys, field[:, :, 0], field[:, :, 1], color='white', density=0.5, arrowsize=1.8, linewidth=1.5)
#     norm_field = np.sqrt(field[:, :, 0] ** 2 + field[:, :, 1] ** 2)
#     ax.pcolor(X, Y, norm_field)
#     return ax


def fixedpoint_task(x0, m, n, hidden_size, I, nojac):
    """
    Task for the root solver to find fixed points, for parallelization
    """
    # Redefining functions for pickle issues
    def F(x, I):
        return -x + m @ (n.T @ np.tanh(x)) / hidden_size + I

    m1 = m[:, 0]
    m2 = m[:, 1]
    n1 = n[:, 0]
    n2 = n[:, 1]
    # Jacobian of F assuming Wrec is rank 2
    def FJac(x, I=None):
        phiPr = phi_prime(x)
        n1_eff = n1 * phiPr
        n2_eff = n2 * phiPr
        return (np.outer(m1, n1_eff) + np.outer(m2, n2_eff)) / hidden_size - np.identity(
            hidden_size
        )

    return root(F, x0, args=I, jac=None if nojac else FJac)


def plot_field_noscalings(
    net,
    vec1=None,
    vec2=None,
    xmin=-3,
    xmax=3,
    ymin=-3,
    ymax=3,
    input=None,
    res=50,
    ax=None,
    add_fixed_points=False,
    fixed_points_trials=10,
    fp_save=None,
    fp_load=None,
    nojac=False,
    orth=False,
    sizes=1.0,
):
    """
    Plot 2d flow field and eventually fixed points for a rank 2 network without scaled vectors (ie defined as in
    Francesca's paper). Can plot the affine flow field in presence of a constant input with argument input.
    :param net: a LowRankRNN
    :param vec1: None or a numpy array of shape (hidden_size). If None, will be taken as vector m1 of the network
    :param vec2: same with m2
    :param xmin: float
    :param xmax: float
    :param ymin: float
    :param ymax: float
    :param input: None or torch tensor of shape (n_inputs), provides constant input for plotting affine flow field
    :param res: int, grid resolution
    :param ax: None or matplotlib axes
    :param add_fixed_points: bool
    :param fixed_points_trials: int, number of simulations to launch to find fixed points
    :param fp_save: None or filename, to save found fixed points instead of plotting them
    :param fp_load: None or filename, to load fixed points instead of recomputing them
    :param nojac: bool, if True, use root solver without jacobian matrix
    :param orth: bool, if True, start by orthogonalizing (vec1, vec2)
    :return: axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    adjust_plot(ax, xmin, xmax, ymin, ymax)
    if vec1 is None:
        vec1 = net.m[:, 0].squeeze().detach().numpy()
    if vec2 is None:
        vec2 = net.m[:, 1].squeeze().detach().numpy()
    if add_fixed_points:
        n1 = net.n[:, 0].squeeze().detach().numpy()
        n2 = net.n[:, 1].squeeze().detach().numpy()
    m = net.m.detach().numpy()
    n = net.n.detach().numpy()

    # Plotting constants
    nx, ny = res, res
    marker_size = 50 * sizes

    # Orthogonalization of the basis vec1, vec2, I
    if orth:
        vec2 = vec2 - (vec2 @ vec1) * vec1 / (vec1 @ vec1)
    if input is not None:
        I = (input @ net.wi_full).detach().numpy()
        I_orth = I - (I @ vec1) * vec1 / (vec1 @ vec1) - (I @ vec2) * vec2 / (vec2 @ vec2)
    else:
        I = np.zeros(net.hidden_size)
        I_orth = np.zeros(net.hidden_size)

    # rescaling factors (for transformation euclidean space / overlap space)
    # here, if one wants x s.t. overlap(x, vec1) = alpha, x should be r1 * alpha * vec1
    # with the overlap being defined as overlap(u, v) = u.dot(v) / sqrt(hidden_size)
    r1 = net.hidden_size / (vec1 @ vec1)
    r2 = net.hidden_size / (vec2 @ vec2)

    # Defining the grid
    xs_grid = np.linspace(xmin, xmax, nx + 1)
    ys_grid = np.linspace(ymin, ymax, ny + 1)
    xs = (xs_grid[1:] + xs_grid[:-1]) / 2
    ys = (ys_grid[1:] + ys_grid[:-1]) / 2
    field = np.zeros((nx, ny, 2))
    X, Y = np.meshgrid(xs, ys)

    # Recurrent function of dx/dt = F(x, I)
    def F(x, I):
        return -x + m @ (n.T @ np.tanh(x)) / net.hidden_size + I

    # Compute flow in each point of the grid
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            h = r1 * x * vec1 + r2 * y * vec2 + I_orth
            delta = F(h, I)
            field[j, i, 0] = delta @ vec1 / (vec1 @ vec1)
            field[j, i, 1] = delta @ vec2 / (vec2 @ vec2)
    ax.streamplot(
        xs,
        ys,
        field[:, :, 0],
        field[:, :, 1],
        color="white",
        density=0.5,
        arrowsize=sizes,
        linewidth=sizes * 0.8,
    )
    norm_field = np.sqrt(field[:, :, 0] ** 2 + field[:, :, 1] ** 2)
    mappable = ax.pcolor(X, Y, norm_field)

    # Look for fixed points
    if add_fixed_points:
        if fp_load is None:
            stable_sols = []
            saddles = []
            sources = []

            # initial conditions are dispersed over a grid
            X_grid, Y_grid = np.meshgrid(
                np.linspace(xmin, xmax, int(sqrt(fixed_points_trials))),
                np.linspace(ymin, ymax, int(sqrt(fixed_points_trials))),
            )

            # Parallelized root solver
            x0s = [
                r1 * X_grid.ravel()[i] * vec1 + r2 * Y_grid.ravel()[i] * vec2 + I_orth
                for i in range(X_grid.size)
            ]
            with mp.Pool(mp.cpu_count()) as pool:
                args = [(x0, m, n, net.hidden_size, I, nojac) for x0 in x0s]
                sols = pool.starmap(fixedpoint_task, args)

            for sol in sols:
                # if solution found
                if sol.success == 1:
                    kappa_sol = [(sol.x @ vec1) / net.hidden_size, (sol.x @ vec2) / net.hidden_size]
                    # Computing stability
                    pseudoJac = np.zeros((2, 2))
                    phiPr = phi_prime(sol.x)
                    n1_eff = n1 * phiPr
                    n2_eff = n2 * phiPr
                    pseudoJac[0, 0] = vec1 @ n1_eff / net.hidden_size
                    pseudoJac[0, 1] = vec2 @ n1_eff / net.hidden_size
                    pseudoJac[1, 0] = vec1 @ n2_eff / net.hidden_size
                    pseudoJac[1, 1] = vec2 @ n2_eff / net.hidden_size
                    eigvals = np.linalg.eigvals(pseudoJac)
                    if np.all(np.real(eigvals) <= 1):
                        stable_sols.append(kappa_sol)
                    elif np.any(np.real(eigvals) <= 1):
                        saddles.append(kappa_sol)
                    else:
                        sources.append(kappa_sol)
        # Load fixed points stored in a file
        else:
            arrays = np.load(fp_load)
            arr = arrays["arr_0"]
            stable_sols = [arr[i] for i in range(arr.shape[0])]
            arr = arrays["arr_1"]
            saddles = [arr[i] for i in range(arr.shape[0])]
            arr = arrays["arr_2"]
            sources = [arr[i] for i in range(arr.shape[0])]
        if fp_save is not None:
            np.savez(fp_save, np.array(stable_sols), np.array(saddles), np.array(sources))
        else:
            ax.scatter(
                [x[0] for x in stable_sols],
                [x[1] for x in stable_sols],
                facecolors="white",
                edgecolors="white",
                s=marker_size,
                zorder=1000,
            )
            ax.scatter(
                [x[0] for x in saddles],
                [x[1] for x in saddles],
                facecolors="black",
                edgecolors="white",
                s=marker_size,
                zorder=1000,
            )
            ax.scatter(
                [x[0] for x in sources],
                [x[1] for x in sources],
                facecolors="black",
                edgecolors="white",
                s=marker_size,
                zorder=1000,
            )
    return ax, mappable


def plot_readout_map(vec1, vec2, wo, rect=(-10, 10, -10, 10), scale=0.5, scalings=True, cmap="jet"):
    """
    Plot the map from the 2D space spanned by (vec1, vec2) to a scalar defined by wo.T @ phi(x)
    :param vec1: numpy array
    :param vec2: numpy array
    :param wo: numpy array
    :param rect: 4-tuple, x and y axis limits
    :param scale: scale of vector readout for plotting
    :param scalings: bool, False for no scalings networks
    :param cmap:
    :return:
    """
    xmin, xmax, ymin, ymax = rect
    hidden_size = vec1.shape[0]
    xs = np.linspace(xmin, xmax, 100)
    ys = np.linspace(ymin, ymax, 100)
    if scalings:
        r1 = 1.0 / (vec1 @ vec1)
        r2 = 1.0 / (vec2 @ vec2)
    else:
        r1 = hidden_size / (vec1 @ vec1)
        r2 = hidden_size / (vec2 @ vec2)
    X, Y = np.meshgrid(xs, ys)
    z = np.zeros((100, 100))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            h = r1 * x * vec1 + r2 * y * vec2
            z[i, j] = wo @ np.tanh(h)
    plt.pcolor(X, Y, z, cmap=cmap)
    plt.colorbar()
    if scalings:
        readout1, readout2 = wo @ vec1, wo @ vec2
    else:
        readout1, readout2 = wo @ vec1 / hidden_size, wo @ vec2 / hidden_size
    plt.quiver(0, 0, readout1, readout2, color="k", scale=scale)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("$\kappa_1$")
    plt.ylabel("$\kappa_2$")
