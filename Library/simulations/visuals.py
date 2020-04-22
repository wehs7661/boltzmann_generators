# Visualization Library
import matplotlib.pyplot as plt
from matplotlib import animation
import potentials
import numpy as np


def make_2D_traj_potential(x_traj, potential, xlim, ylim, min = -10, max = -3, fps = 30, markersize = 8, color = 'red'):
    X, Y = np.meshgrid(np.linspace(xlim[0],xlim[1],50), np.linspace(ylim[0],ylim[1],50))
    Z = []
    for i in range(len(X)):
        z = []
        for j in range(len(X)):
            v = potential([X[i,j], Y[i,j]])
            z.append(v)
        Z.append(z)
    Z = np.array(Z)

    fps = fps
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contourf(X,Y,Z, levels = 500, cmap='jet')
    sct, = ax.plot([], [], "o", markersize=markersize, color = color, vmin = min, vmax = max)
    cbar = fig.colorbar(cs)
    # pct, = ax.plot([], [], "o", markersize=8)

    def update_graph(i, xa, ya): # , vx, vy):
        sct.set_data(xa[i], ya[i])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    video_traj = x_traj
    ani = animation.FuncAnimation(fig, update_graph, video_traj.shape[0], fargs=(video_traj[:,:,0], video_traj[:,:,1]), interval=1000/fps)
    plt.rcParams['animation.html'] = 'html5'
    return(ani)


def make_2D_traj(x_traj, box, fps = 30, markersize = 8, color = 'red'):
    fps = fps
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sct, = ax.plot([], [], "o", markersize=markersize, color = color)
    # pct, = ax.plot([], [], "o", markersize=8)

    def update_graph(i, xa, ya): # , vx, vy):
        sct.set_data(xa[i], ya[i])

    x_lim = box[0]/2 
    y_lim = box[1]/2
    ax.set_xlim([-x_lim, x_lim])
    ax.set_ylim([-y_lim, y_lim])
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    video_traj = x_traj
    ani = animation.FuncAnimation(fig, update_graph, video_traj.shape[0], fargs=(video_traj[:,:,0], video_traj[:,:,1]), interval=1000/fps)
    plt.rcParams['animation.html'] = 'html5'
    return(ani)

def plot_2D_potential(potential, xlim, ylim, min = -10, max = -3, fps = 30, markersize = 8, color = 'red'):
    X, Y = np.meshgrid(np.linspace(xlim[0],xlim[1],50), np.linspace(ylim[0],ylim[1],50))
    Z = []
    for i in range(len(X)):
        z = []
        for j in range(len(X)):
            v = potential([X[i,j], Y[i,j]])
            z.append(v)
        Z.append(z)
    Z = np.array(Z)
    cs = plt.contourf(X,Y,Z, levels = 500, cmap='jet', vmin=min, vmax=max)
    cbar = plt.colorbar(cs)
    plt.ylabel("$x_2$")
    plt.xlabel("$x_1$")