"""Utilities for working with plots and tf events."""
from matplotlib.lines import Line2D
from stable_baselines3.common.base_class import BaseAlgorithm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os, shutil, logging
from matplotlib import animation
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib import colors as clr
from numpy import pi, sin, cos
import pandas as pd


logger = logging.getLogger(__name__)

# COLORS_PLOTS = [
#     '#0065bd', #blue
#     '#e37222', #orange
#     '#008f7c', #green
#     '#f0266b' #magenta
# ]


COLORS_PLOTS = [
    'tab:blue', #blue
    'tab:orange', #orange
    'tab:green', #green
    '#f0266b' #magenta
]

# https://gist.github.com/lnksz/51e3566af2df5c7aa678cd4dfc8305f7
COLORS = {
    'BLUE': (0 / 255, 101 / 255, 189 / 255),
    'ORANGE': (227 / 255, 114 / 255, 34 / 255),
    'LIGHT_BLUE': (100 / 255, 160 / 255, 200 / 255),
    'LIGHTER_BLUE': (152 / 255, 198 / 255, 234 / 255)
}
COLORS_HEX = {
    'BLUE': '#0065bd',
    'ORANGE': '#e37222',
    'LIGHT_BLUE': '#64a0c8',
    'LIGHTER_BLUE': '#98c6ea'
}

def save_model(name, model):
    path = os.getcwd() + '/models/'
    os.makedirs(path, exist_ok=True)
    model.save(path + name)  # TODO: Check if save_to_zip_file

def load_model(name, base: BaseAlgorithm):
    path = os.getcwd() + '/models/'
    if os.path.isfile(path + name):
        return base.load(path + name)
    else:
        raise FileNotFoundError(f'No such model {name}')

def gain_matrix():
    """ Gain matrix (K) of the pendulum's LQR controller."""
    import matlab.engine
    path_matlab = f'{os.path.dirname(os.path.abspath(__file__))}/matlab'
    eng = matlab.engine.start_matlab()
    path_matlab = eng.genpath(path_matlab)
    eng.addpath(path_matlab, nargout=0)
    K = eng.gainMatrix()
    eng.quit()
    return K

def torque_given_state(gain_matrix, state):
    return np.dot(gain_matrix, state)

def setup_plot(fig, width, height, font_size=11):
    """ Used to initialize uniform plots.
    """
    fig.set_size_inches(width, height)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = font_size
    # plt.gca().tick_params(axis="x", direction="in")
    # plt.gca().tick_params(axis="y", direction="in")

def finalize_plot(fig=None, width=.0, height=.0, x_label='', y_label='', path=None):
    """ Finalize and save plots.
    """
    plt.gca().tick_params(axis="x", direction="out", zorder=10)  # =0)
    plt.gca().tick_params(axis="y", direction="out", zorder=10)  # length=0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if path:
        plt.savefig(path, dpi=1000, bbox_inches='tight')
    plt.show()



def safety_measure_plot(v1, v2, width=2.5, height=2.5, l=1, m=1, g=9.81, K=None, vertices=None, save_as=None):
    # Custom starting points for clean/useful plots
    # start_points = [
    #     [pi / 2, 0],
    #     [-pi / 2, 0],
    #     [0, -2 * pi],
    #     [0, 2 * pi],
    #     [-0.55 * pi, -4 * pi],
    #     [0.55 * pi, 4 * pi],
    #     [-pi, -4 * pi],
    #     [pi, 4 * pi],
    #     [-1.5 * pi, 2 * pi],
    #     [1.5 * pi, -2 * pi],
    #     [-0.8 * pi, 4 * pi],
    #     [0.8 * pi, -4 * pi],
    # ]

    # theta, thetadot = np.meshgrid(np.linspace(-1.5 * np.pi, 1.5 * np.pi, 375), np.linspace(-5 * np.pi, 5 * np.pi, 350))
    # u = np.dot(np.moveaxis([theta, thetadot], 0, -1), K)
    # thetadotdot = g / l * sin(theta) - (1 / (m * l ** 2)) * u

    fig = plt.figure()
    setup_plot(fig=fig, width=width, height=height)
    linewidth = 1

    vertices3 = [[-6, 16], [6, 16], [6, -16], [-6, -16], [0,0]]
    codes = [Path.MOVETO]
    for _ in range(len(vertices3) - 2):
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    path = Path(vertices3, codes)
    polytope = patches.PathPatch(path,
                                 facecolor='red',
                                 edgecolor='none',
                                 linewidth=linewidth,
                                 linestyle='-',  # {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
                                 alpha=0.1,
                                 zorder=0)

    plt.gca().add_patch(polytope)

    if vertices is not None:
        codes = [Path.MOVETO]
        for _ in range(len(vertices) - 1):
            codes.append(Path.LINETO)
        codes.append(Path.CLOSEPOLY)
        vertices = np.vstack((vertices, [0., 0.]))
        path = Path(vertices, codes)
        polytope0 = patches.PathPatch(path,
                                     facecolor='white',
                                     edgecolor='none',
                                     linewidth=linewidth,
                                     linestyle='-',  # {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
                                     alpha=1,
                                     zorder=1)
        polytope = patches.PathPatch(path,
                                      facecolor='green',
                                      edgecolor='magenta',
                                      linewidth=linewidth,
                                      linestyle='-',  # {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
                                      alpha=0.1,
                                      zorder=2)
        polytope2 = patches.PathPatch(path,
                                     facecolor='none',
                                     edgecolor='magenta',
                                     linewidth=linewidth,
                                     linestyle='-',  # {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
                                     alpha=1,
                                     zorder=3)
        plt.gca().add_patch(polytope0)
        plt.gca().add_patch(polytope)
        plt.gca().add_patch(polytope2)

    # plt.gca().scatter(.8*v2[0],.8*v2[1]-1.5, s=8, c='black', zorder=4)
    # plt.gca().scatter(.55*v2[0] - .3,.55*v2[1]-1.5, s=4, c='green', zorder=4)
    # plt.gca().scatter(.55*v2[0] - .3,.55*v2[1]+1.5, s=4, c='green', zorder=4)
    # plt.gca().scatter(.55*v2[0] + .3,.55*v2[1]-1.5, s=4, c='green', zorder=4)
    # plt.gca().scatter(.55*v2[0] + .3,.55*v2[1]+1.5, s=4, c='green', zorder=4)
    #
    # plt.gca().scatter(0.55 * v2[0], 0.55 * v2[1], s=8, c='green', zorder=4)
    # plt.gca().annotate("", color="black", xy=(.55*v2[0],.55*v2[1]), xytext=(.8*v2[0],.8*v2[1]-1.5),
    #                     arrowprops=dict(arrowstyle="-|>", mutation_scale=5, color="black"), annotation_clip=True, zorder=6)
    # rect = patches.Rectangle((0.55 * v2[0]-.3, 0.55 * v2[1]-1.5), .6, 3, linewidth=.75, edgecolor='blue', facecolor='none', zorder=3)
    # plt.gca().add_patch(rect)

    # plt.gca().scatter(.35*v2[0], .35*v2[1], s=8, c='black', zorder=7)
    # plt.gca().scatter(-.25, 5.25, s=8, c='red', zorder=4)
    # #plt.gca().scatter(.2 * v2[0] +.3, .2 * v2[1] + 3 -1.5 -.2, s=4, c='green', zorder=4)
    # #plt.gca().scatter(.2 * v2[0] +.3, .2 * v2[1] + 3 +1.5 -.2, s=4, c='red', zorder=4)
    # #plt.gca().scatter(.2 * v2[0] -.3, .2 * v2[1] + 3 -1.5 -.2, s=4, c='green', zorder=4)
    # #plt.gca().scatter(.2 * v2[0] -.3, .2 * v2[1] + 3 +1.5 -.2, s=4, c='green', zorder=4)
    # #rect = patches.Rectangle((.2*v2[0] -.3, .2*v2[1] + 3 -1.5 -.2), .6, 3, linewidth=.75, edgecolor='blue',
    # #                         facecolor='none', zorder=3)
    # #plt.gca().add_patch(rect)
    # plt.gca().scatter(.15*v2[0] - .2, -1.25 + .15*v2[1], s=6, c='green', zorder=5)
    # plt.gca().scatter(.15*v2[0] - .2, +1.25 + .15*v2[1] , s=6, c='green', zorder=5)
    # plt.gca().scatter(.15*v2[0] +.2, +1.25 + .15*v2[1] , s=6, c='green', zorder=5)
    # plt.gca().scatter(.15*v2[0] +.2, -1.25 + .15*v2[1] , s=6, c='green', zorder=5)



    # plt.gca().annotate("", color="red", xy=(-.25, 5.25),
    #                    xytext=(.35*v2[0], .35*v2[1]),
    #                    arrowprops=dict(arrowstyle="-|>", mutation_scale=5, color="red"), annotation_clip=True,
    #                    zorder=6)
    # plt.gca().annotate("", color="green", xy=(.15*v2[0], .15*v2[1]),
    #                    xytext=(.35*v2[0], .35*v2[1]),
    #                    arrowprops=dict(arrowstyle="-|>", mutation_scale=5, color="green"), annotation_clip=True,
    #                    zorder=5)
    #
    #
    #
    # rect = patches.Rectangle((-.2 + .15*v2[0], -1.25 + .15*v2[1]), .4, 2.5, linewidth=1,
    #                          edgecolor='blue',
    #                          facecolor='none', zorder=4)
    # plt.gca().add_patch(rect)
    # plt.gca().scatter(.15*v2[0], .15*v2[1], s=8, c='black', zorder=4)
    #plt.gca().scatter(- .3, - 1.5, s=4, c='green', zorder=6)
    #plt.gca().scatter(- .3,+1.5, s=4, c='green', zorder=6)
    #plt.gca().scatter(.3, -1.5, s=4, c='green', zorder=6)
    #plt.gca().scatter(.3, +1.5, s=4, c='green', zorder=6)

    #plt.gca().scatter(pi/2 +.25*v2[0], -2.25*pi -.3 +.25*v2[1], s=8, c='black', zorder=8)
    #plt.gca().scatter(pi / 2 - .2*v2[0] +.2*v2[0], -2.25 * pi - .2*v2[1] + .5 -.3 +.2*v2[1], s=8, c='red', zorder=4)

    #plt.gca().scatter(pi / 2 - .2*v2[0] -.3, -2.25 * pi - .2*v2[1] + .5 -.3 -1.5, s=4, c='red', zorder=4)
    #plt.gca().scatter(pi / 2 - .2 * v2[0] -.3, -2.25 * pi - .2 * v2[1] + .5 - .3 +1.5, s=4, c='green', zorder=4)
    #plt.gca().scatter(pi / 2 - .2 * v2[0] +.3, -2.25 * pi - .2 * v2[1] + .5 - .3 -1.5, s=4, c='green', zorder=4)
    #plt.gca().scatter(pi / 2 - .2 * v2[0] +.3, -2.25 * pi - .2 * v2[1] + .5 - .3 +1.5, s=4, c='green', zorder=4)

    # plt.gca().annotate("", color="red", xy=(pi / 2 + .27 * v2[0] + .2 +.27*v2[0], -2.15 * pi + .27 * v2[1] - 1.2 +.27*v2[1]),
    #                    xytext=(pi/2 +.25*v2[0], -2.25*pi -.3 +.25*v2[1]),
    #                    arrowprops=dict(arrowstyle="-|>", mutation_scale=5, color="green"), annotation_clip=True,
    #                    zorder=5)
    #
    # plt.gca().annotate("", color="orange", xy=(pi / 2 + .27 * v2[0] +.04 +.25*v2[0], -2.25 * pi + .27 * v2[1] -1.85 +.25*v2[1]),
    #                    xytext=(pi/2 +.25*v2[0], -2.25*pi -.3 +.25*v2[1]),
    #                    arrowprops=dict(arrowstyle="-|>", mutation_scale=5, color="orange"), annotation_clip=True,
    #                    zorder=6)
    #
    # plt.gca().annotate("", color="red",
    #                    xy=(1.45, 0),
    #                    xytext=(pi/2 +.25*v2[0], -2.25*pi -.3 +.25*v2[1]),
    #                    arrowprops=dict(arrowstyle="-|>", mutation_scale=5, color="red"), annotation_clip=True,
    #                    zorder=6)


    #rect = patches.Rectangle((pi / 2 - .2*v2[0] -.3, -2.25 * pi - .2*v2[1] + .5 -1.5 -.3), .6, 3, linewidth=.75, edgecolor='blue',
    #                         facecolor='none', zorder=3)
    # plt.gca().add_patch(rect)
    #
    # plt.gca().scatter(1.45, 0, s=8,
    #                   c='red', zorder=4)


    # plt.gca().scatter(pi / 2 + .27 * v2[0] +.04 +.25*v2[0], -2.25 * pi + .27 * v2[1] -1.85 +.25*v2[1], s=8, c='orange', zorder=4)
    # plt.gca().scatter(pi / 2 + .27 * v2[0] + .2 +.27*v2[0], -2.15 * pi + .27 * v2[1] - 1.2 +.27*v2[1], s=8, c='black', zorder=4)
    # #plt.gca().scatter(pi / 2 + .2*v2[0] + .05, -2.25 * pi + .2*v2[1] -.15, s=8, c='black', zorder=4)
    #
    # plt.gca().scatter(pi / 2 + .27 * v2[0] +.04 -.3 +.25*v2[0], -2.25 * pi + .27 * v2[1] -1.85 -2 +.25*v2[1], s=6, c='red', zorder=4)
    # plt.gca().scatter(pi / 2 + .27 * v2[0] +.04 +.3 +.25*v2[0], -2.25 * pi + .27 * v2[1] -1.85 -2 +.25*v2[1], s=6, c='red', zorder=4)
    # plt.gca().scatter(pi / 2 + .27 * v2[0] +.04 -.3 +.25*v2[0], -2.25 * pi + .27 * v2[1] -1.85 +2 +.25*v2[1], s=6, c='green', zorder=4)
    # plt.gca().scatter(pi / 2 + .27 * v2[0] +.04 +.3 +.25*v2[0], -2.25 * pi + .27 * v2[1] -1.85 +2 +.25*v2[1], s=6, c='green', zorder=4)
    #
    # rect = patches.Rectangle((pi / 2 + .27 * v2[0] +.04 -.3 +.25*v2[0], -2.25 * pi + .27 * v2[1] -1.85 -2 +.25*v2[1]), .6, 4, linewidth=1,
    #                          edgecolor='blue',
    #                          facecolor='none', zorder=3)
    # plt.gca().add_patch(rect)




    # plt.gca().annotate("$s_{\mathrm{t}}^{1}$", xytext=(.35*v2[0] -.28, .35*v2[1] -.2), xy=(0, 0))
    # plt.gca().annotate("$s_{\mathrm{t}}^{2}$", xytext=(pi/2 +.25*v2[0] + .1, -2.25*pi -.3 +.25*v2[1] -.6), xy=(0, 0))
    # plt.gca().annotate("$a_{\mathrm{t}}^{\mathrm{RL}}$", xytext=(pi / 2 + .25 * v2[0] + .1 -.1*v2[0] - .1, -2 -.1*v2[1] - .5),
    #                    xy=(0, 0))
    # plt.gca().annotate("$a_{\mathrm{t}}^{\mathrm{RL}}$", xytext=(.3*v2[0]-.125, .3*v2[1]+1.6),
    #                    xy=(0, 0))
    # plt.gca().annotate("$a_{\mathrm{t}}^{\mathrm{VER}}$", xytext=(pi / 2 + .25 * v2[0] + .1 -.5, -3.5),
    #                    xy=(0, 0))
    # plt.gca().annotate("$a_{\mathrm{t}}^{\mathrm{VER}}$", xytext=(.3*v2[0] -.28 -.07, .3*v2[1] -.2 -1.55),
    #                    xy=(0, 0))

    #plt.gca().annotate("$s_{\mathrm{t}}^{1}$", xytext=(.8*v2[0] -.17,.8*v2[1]-1.5 + 1.1), xy=(0, 0))
    #plt.gca().annotate("$s_{\mathrm{t}+1}^{1}$", xytext=(0.35 * v2[0] + .4, 0.35 * v2[1] - .9), xy=(0, 0))

    #plt.gca().annotate("$s_{\mathrm{t}}^{2}$", xytext=(-.6*v2[0] - .05 - 3,-.6*v2[1] - 2 + 7.9), xy=(0, 0))
    #plt.gca().annotate("$s_{\mathrm{t}}^{3}$", xytext=(pi/2 - .2, -2*pi + .075), xy=(0, 0))
    #plt.gca().annotate("$s_{\mathrm{t}+1}^{2}$", xytext=(-0.45 * v2[0] - 0.3 + .1, -0.25 * v2[1] + 2.5), xy=(0, 0))

    vertices = [
        [0.4 * (v1[0] + v2[0]), 0.4 * (v1[1] + v2[1])],
        [0.4 * (v1[0] - v2[0]), 0.4 * (v1[1] - v2[1])],
        [0.4 * (-v1[0] - v2[0]), 0.4 * (-v1[1] - v2[1])],
        [0.4 * (-v1[0] + v2[0]), 0.4 * (-v1[1] + v2[1])]
    ]

    if vertices is not None:
        codes = [Path.MOVETO]
        for _ in range(len(vertices) - 1):
            codes.append(Path.LINETO)
        codes.append(Path.CLOSEPOLY)
        vertices = np.vstack((vertices, [0., 0.]))
        path = Path(vertices, codes)
        polytope = patches.PathPatch(path,
                                     facecolor='none',
                                     edgecolor='green',
                                     linewidth=linewidth,
                                     linestyle='-',  # {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
                                     alpha=1.0,
                                     zorder=3)
        plt.gca().add_patch(polytope)


    vertices = [
        [1.6 * (v1[0] + v2[0]), 1.6 * (v1[1] + v2[1])],
        [1.6 * (v1[0] - v2[0]), 1.6 * (v1[1] - v2[1])],
        [1.6 * (-v1[0] - v2[0]), 1.6 * (-v1[1] - v2[1])],
        [1.6 * (-v1[0] + v2[0]), 1.6 * (-v1[1] + v2[1])]
    ]

    if vertices is not None:
        codes = [Path.MOVETO]
        for _ in range(len(vertices) - 1):
            codes.append(Path.LINETO)
        codes.append(Path.CLOSEPOLY)
        vertices = np.vstack((vertices, [0., 0.]))
        path = Path(vertices, codes)
        polytope = patches.PathPatch(path,
                                     facecolor='none',
                                     edgecolor='red',
                                     linewidth=linewidth,
                                     linestyle='-',  # {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
                                     alpha=1.0,
                                     zorder=3)
        plt.gca().add_patch(polytope)

    # plt.arrow(0, 0, 0.25*v2[0], 0.25*v2[1], zorder=2, length_includes_head=True)
    # plt.arrow(0,0, -1, -0.6*v1[1], zorder=2, length_includes_head=True)
    #
    # plt.gca().annotate("", xy=(-0.3*v2[0], -0.3*v2[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", mutation_scale=5, color="black"), zorder=3)
    # plt.gca().annotate("", xy=(-0.5 * v1[0], -0.5 * v1[1]), xytext=(0, 0),
    #                   arrowprops=dict(arrowstyle="-|>", mutation_scale=5, color="black"), zorder=3)
    #
    # plt.gca().annotate("", xy=(-1.2*v2[0], -1.2*v2[1]), xytext=(0, 0),
    #                   arrowprops=dict(arrowstyle="-|>", mutation_scale=5, color="black"), zorder=3)
    # plt.gca().annotate("", xy=(1.4*v1[0] -1.2*v2[0], 1.4*v1[1] -1.2*v2[1]),
    #                   xytext=(-1.2*v2[0], -1.2*v2[1]),
    #                   arrowprops=dict(arrowstyle="-|>", mutation_scale=5, color="black"), zorder=3)

    plt.gca().annotate("$v$", xytext=(-.03*v2[0], -.03*v2[1]-1.2),xy=(0, 0))
    plt.gca().annotate("$w$", xytext=(-1.7+.15*v2[0], 5.5+.15*v2[1]-.15), xy=(0, 0))
    plt.gca().annotate("", color="magenta", xy=(v1[0], v1[1]), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", mutation_scale=5, color="black"), annotation_clip=True, zorder=4)
    plt.gca().annotate("", color="magenta", xy=(v2[0],v2[1]), xytext=(0, 0), arrowprops = dict(arrowstyle="-|>",
                                                                     mutation_scale=5, color="black"), zorder=4)

    #safe = Line2D([0], [0], color="green", label='$\mathrm{Safe}$')
    #unsafe = Line2D([0], [0], color="red", label='$$\mathrm{Unsafe}$$')
    #plt.legend(loc="upper right", handles=[safe, unsafe], handlelength=0)

    #length = Line2D([0], [0], color="black", label='$\left|v_{1}\\right|=\left|v_{2}\\right|=1$')
    #plt.legend(loc="upper right", handles=[length], markerscale=0,  handlelength=0, handletextpad=-.05)
    #leg = plt.gca().get_legend()
    #leg.legendHandles[0].set_color('none')

    #length = Line2D([0], [0], color="black", label='$\max\limits_{i\in\{1,2\}}|\\alpha_{i}|$')
    length = Line2D([0], [0], color="black", label='$\max\left(|\\alpha|,|\\beta|\\right)$')
    leg = plt.legend(loc="upper right", handles=[length], markerscale=0,  handlelength=0, handletextpad=-.05)
    leg.legendHandles[0].set_color('none')
    safe = Line2D([0], [0], color="green", label='$0.4$')
    boundary = Line2D([0], [0], color="magenta", label='$1.0$')
    unsafe = Line2D([0], [0], color="red", label='$1.6$')
    plt.legend(loc="lower left", handles=[safe, boundary, unsafe], handlelength=1.5)
    plt.gca().add_artist(leg)

    #arrow = plt.arrow(0, 0, 0.5, 0.6, 'dummy', label='$a_{t}^{\mathrm{VER}}$', color="orange")
    #plt.legend([arrow, ], ['My label', ])

    #euler = Line2D([0], [0], color="blue", label='$\pm$LTE')
    #unsafe = Line2D([0], [0], color="red", label='Unsafe')
    #rlver = Line2D([0], [0], color="orange", label='$a_{\mathrm{t}}^{\mathrm{VER}}$')
    #rl = Line2D([0], [0], color="black", label='$s_{\mathrm{t}(+1)}^{\mathrm{i}}$, $a_{\mathrm{t}}^{\mathrm{i}}$')
    #leg = plt.legend(loc="lower left", handles=[rlver, rl], handlelength=.8)
    #plt.legend(loc="upper right", handles=[euler, unsafe], handlelength=.8)
    #plt.legend(loc="upper right", handles=[euler], handlelength=.8)
    #plt.gca().add_artist(leg)


    #plt.gca().set_aspect('equal', adjustable='box')

    #arrowstyle = "-|>",
    #arrowsize = 0.65


    # color = (abs(theta) + (1.5 / 4) * abs(thetadot)) ** 0.5
    # cmap = clr.LinearSegmentedColormap.from_list("", [COLORS_HEX['BLUE'], COLORS_HEX['ORANGE']])

    # plt.streamplot(theta, thetadot, thetadot, thetadotdot,
    #                density=30,
    #                linewidth=linewidth,
    #                cmap=cmap,
    #                color=color,
    #                start_points=start_points,
    #                )
    for c in plt.gca().get_children():
        if not isinstance(c, patches.FancyArrowPatch):
            continue
        c.remove()

    equilibrium = np.array([[0], [0]])
    plt.gca().scatter(equilibrium[0], equilibrium[1], s=8, c=COLORS_HEX['BLUE'], zorder=4)

    plt.yticks([-4 * np.pi, -2 * np.pi, 0, 2 * np.pi, 4 * np.pi])
    plt.gca().set_yticklabels(["$-4\pi$", "$-2\pi$", "0", "$2\pi$", "$4\pi$"])
    plt.xticks([-np.pi, 0, np.pi])
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    #plt.xlim([-pi/2, pi/2])
    plt.xlim([-1.5 * pi, 1.5 * pi])
    #plt.ylim([-2.25 * pi, 2.25 * pi])
    plt.ylim([-5 * pi, 5 * pi])
    plt.gca().set_xticklabels(
        ["$-\pi$", "$-\\frac{\pi}{2}$", "0", "$\\frac{\pi}{2}$", "$\pi$"])

    plt.gca().xaxis.grid(True, linestyle='dotted', linewidth=0.5)
    plt.gca().yaxis.grid(True, linestyle='dotted', linewidth=0.5)
    # plt.gca().axhline(0, color='gray', linewidth=0.5)

    if save_as:
        finalize_plot(x_label='$\\theta[\mathrm{rad}]$',
                      y_label='$\dot{\\theta}[\\frac{\mathrm{rad}}{\mathrm{s}}]$',
                      path=f'{os.getcwd()}/{save_as}.pdf')
        # path=f'{os.getcwd()}/../../report/thesis/figures/{save_as}.pdf')
    else:
        finalize_plot(x_label='$\\theta[\mathrm{rad}]$',
                      y_label='$\dot{\\theta}[\\frac{\mathrm{rad}}{\mathrm{s}}]$')

def phase_plot(width=2.5, height=2.5, l=1, m=1, g=9.81, K=None, max_torque=None, max_theta=None, vertices=None, boxes=None,
               save_as=None):
    """ Generates (and saves) a phase plot of a (controlled) mathematical / simple gravity pendulum."""


    theta, thetadot = np.meshgrid(np.linspace(-1.5 * np.pi, 1.5 * np.pi, 375), np.linspace(-5 * np.pi, 5 * np.pi, 350))

    if K is None:

        start_points = [
            [0, 4 * pi],
            [0, -4 * pi],
            [0, 3 * pi],
            [0, -3 * pi],
            [0, 2 * pi],
            [0, -2 * pi],
            [0, 1 * pi],
            [0, -1 * pi],
            [-pi, 1.7*pi],
            [pi, 1.7 * pi],
            [-pi, 1.2 * pi],
            [pi, 1.2 * pi],
            [-pi, 0.7 * pi],
            [pi, 0.7 * pi],
        ]
        # Uncontrolled system
        thetadotdot = g / l * sin(theta)
        color =  (abs(np.sin(theta)) + (1.5/4)*abs(thetadot)) ** 0.5
        cmap = clr.LinearSegmentedColormap.from_list("", [COLORS_HEX['BLUE'], COLORS_HEX['ORANGE']])

    else:

        # Custom starting points for clean/useful plots
        start_points = [
            [pi/2, 0],
            [-pi/2, 0],
            [0, -2*pi],
            [0, 2 * pi],
            [-0.55*pi, -4*pi],
            [0.55*pi, 4*pi],
            [-pi, -4 * pi],
            [pi, 4 * pi],
            [-1.5*pi, 2*pi],
            [1.5 * pi, -2 * pi],
            [-0.8*pi, 4*pi],
            [0.8 * pi, -4 * pi],


        ]

        # Controlled system
        if max_torque and max_theta:

            u = np.dot(np.moveaxis([theta, thetadot], 0, -1), K)
            thetadotdot = g / l * sin(theta) - (1 / (m * l ** 2)) * u

            orange = (abs(np.dot(np.moveaxis([theta, thetadot], 0, -1), K)) <= max_torque)
            orange2 = (abs(theta) > max_theta)#.astype(int)
            orange = -10000*np.logical_and(orange, orange2).astype(int)
            green1 = (abs(np.dot(np.moveaxis([theta, thetadot], 0, -1), K)) <= max_torque)
            green2 = (abs(theta) <= pi) #Artefacts in plot
            green3 = (abs(theta) <= max_theta)
            green = np.logical_and(green1, green2)
            green = 10000*np.logical_and(green, green3).astype(int)
            color = orange + green
            cmap = clr.ListedColormap(['orange', 'red', 'green'])

        else:

            thetadotdot = g / l * sin(theta) - (1 / (m * l ** 2)) * np.dot(np.moveaxis([theta, thetadot], 0, -1), K)
            color = (abs(theta) + (1.5/4)*abs(thetadot)) ** 0.5
            cmap = clr.LinearSegmentedColormap.from_list("", [COLORS_HEX['BLUE'], COLORS_HEX['ORANGE']])

    fig = plt.figure()
    setup_plot(fig=fig, width=width, height=height)
    linewidth = 1

    if vertices is not None:
        codes = [Path.MOVETO]
        for _ in range(len(vertices) - 1):
            codes.append(Path.LINETO)
        codes.append(Path.CLOSEPOLY)
        vertices = np.vstack((vertices, [0., 0.]))
        path = Path(vertices, codes)
        polytope = patches.PathPatch(path,
                                     facecolor='none',
                                     edgecolor='magenta',
                                     linewidth=linewidth,
                                     linestyle='-',  # {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
                                     alpha=1.0,
                                     zorder=3)
        plt.gca().add_patch(polytope)

    if boxes is not None:

        for b in boxes:
            w, h = b[1] - b[0]

            #if abs(b[1,1]) <= max_thdot and abs(b[0,1]) <= max_thdot:
            box = patches.Rectangle(b[0], w, h,
                                    facecolor='none',
                                    edgecolor='darkturquoise',
                                    linewidth=.85,
                                    linestyle='-',  # {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
                                    alpha=1.0,
                                    zorder=2)
            plt.gca().add_patch(box)

    if vertices is None and boxes is None:
        plt.streamplot(theta, thetadot, thetadot, thetadotdot,
                       density=30,
                       linewidth=linewidth,
                       cmap=cmap,
                       arrowstyle="-|>",
                       arrowsize=0.65,
                       start_points=start_points,
                       color=color,
                       )
    else:
        plt.streamplot(theta, thetadot, thetadot, thetadotdot,
                       density=30,
                       linewidth=linewidth,
                       cmap=cmap,
                       start_points=start_points,
                       color=color,
                       )
        for c in plt.gca().get_children():
            if not isinstance(c, patches.FancyArrowPatch):
                continue
            c.remove()

    if K is None:
        equilibrium = np.array([[-np.pi, 0, np.pi], [0, 0, 0]])
    else:
        equilibrium = np.array([[0], [0]])

    plt.gca().scatter(equilibrium[0], equilibrium[1], s=8, c=COLORS_HEX['BLUE'], zorder=4)

    plt.yticks([-4* np.pi, -2 * np.pi, 0, 2 * np.pi, 4*np.pi])
    plt.gca().set_yticklabels(["$-4\pi$", "$-2\pi$", "0", "$2\pi$", "$4\pi$"])
    plt.xticks([-np.pi, 0, np.pi])
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    plt.xlim([-1.5*pi, 1.5*pi])
    plt.ylim([-5*pi, 5*pi])
    plt.gca().set_xticklabels(
        ["$-\pi$", "$-\\frac{\pi}{2}$", "0", "$\\frac{\pi}{2}$", "$\pi$"])

    plt.gca().xaxis.grid(True, linestyle='dotted', linewidth=0.5)
    plt.gca().yaxis.grid(True, linestyle='dotted', linewidth=0.5)
    #plt.gca().axhline(0, color='gray', linewidth=0.5)

    #plt.gca().yaxis.set_label_position("right")
    #plt.gca().yaxis.tick_right()

    if save_as:
        finalize_plot(x_label='$\\theta[\mathrm{rad}]$',
                      y_label='$\dot{\\theta}[\\frac{\mathrm{rad}}{\mathrm{s}}]$',
                      path = f'{os.getcwd()}/{save_as}.pdf')
                      #path=f'{os.getcwd()}/../../report/thesis/figures/{save_as}.pdf')
    else:
        finalize_plot(x_label='$\\theta[\mathrm{rad}]$',
                      y_label='$\dot{\\theta}[\\frac{\mathrm{rad}}{\mathrm{s}}]$')


def external_legend_res(labels, save_as=None, width=1., height=.25, ncols=2, colors=None, equi=False):
    fig = plt.figure()
    setup_plot(fig=fig, width=width, height=height)
    plt.gca().axis('off')

    handles = []
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i, label in enumerate(labels):
        color = colors[i]
        handles.append(Line2D([0], [0], color=color, label=label))

    if equi:
        handles.append(Line2D([0], [0], color="white", markerfacecolor=COLORS_HEX['BLUE'], marker='o', label='Equilibrium'))

    #, ncol=2
    plt.gca().legend(handles=handles, frameon=True, loc='center', ncol=ncols)

    if save_as is not None:
        plt.savefig(f'{os.getcwd()}/{save_as}.pdf', dpi=1000, bbox_inches='tight')
    plt.show()


def external_legend(save_as=None, width=1., height=.25):

    fig = plt.figure()
    setup_plot(fig=fig, width=width, height=height)
    plt.gca().axis('off')

    b_line = Line2D([0], [0], color="white", markerfacecolor=COLORS_HEX['BLUE'], marker='o', label='Equilibrium')
    m_line = Line2D([0], [0], color='magenta', label='ROA (Polygon)')
    g_line = Line2D([0], [0], color='green', label='$|\\tau|\\leq\\tau_{\mathrm{max}}\;\mathrm{and}\;|\\theta|\\leq\\theta_{\mathrm{max}}$')
    o_line = Line2D([0], [0], color='orange', label='$|\\tau|\\leq\\tau_{\mathrm{max}}\;\mathrm{and}\;|\\theta|>\\theta_{\mathrm{max}}$')
    r_line = Line2D([0], [0], color='red', label='$|\\tau| >\\tau_{\mathrm{max}}$')
    t_line = Line2D([0], [0], color='darkturquoise', label='ROA (Subpaving)')

    plt.gca().legend(handles=[g_line, o_line, r_line, b_line, t_line, m_line], frameon=True, loc='center', ncol=2)

    if save_as is not None:
        plt.savefig(f'{os.getcwd()}/{save_as}.pdf', dpi=1000, bbox_inches='tight')
    plt.show()

def remove_tf_logs(*dirs):
    """ Removes directories within ./tensorboard.
    """
    cwd = os.getcwd() + '/tensorboard/'
    if not dirs:
        dirs = os.listdir(cwd)
    for dir in dirs:
        shutil.rmtree(cwd + dir, ignore_errors=True)


def rename_tf_events(group):
    """ Renames the tf events within ./tensorboard/*dirs to tfevents.event.
    """
    cwd = os.getcwd() + f"/tensorboard/{group}/"

    dirs = [dir for dir in os.listdir(cwd) if os.path.isdir(cwd + dir)]
    for dir in dirs:
        files = os.listdir(cwd + dir + '/')
        if len(files) != 1:
            logger.warning(f'No unique event file found within {cwd + dir}')
        os.rename(cwd + dir + '/' + files[0],
                  cwd + dir + '/' + 'tfevents.event')




def smooth_data(ys, window_size=5):
    if window_size <= 0 or window_size % 2 != 1:
        raise ValueError('Choose window_size > 0 and window_size % 2 == 1')
    w = np.ones(window_size)
    # Adapted from https://github.com/openai/spinningup/blob/master/spinup/utils/plot.py
    return np.convolve(ys, w, mode='same') / np.convolve(np.ones(len(ys)), w, mode='same')

def tf_events_to_plot(dirss, tags, x_label='Episode', y_label='', width=5, height=2.5, episodes =20e2, episode_length=100, window_size=1, save_as=None):

    cwd = os.getcwd() + '/tensorboard/'

    fig = plt.figure()
    setup_plot(fig=fig, width=width, height=height)


    if not tags:
        logger.error(f'No tags specified: {tags}')
        return

    else:

        x_min = None
        x_max = None

        safety_only = True

        for tag in tags:


            for dirs in dirss:

                label = None
                if isinstance(dirs, str):
                    label = dirs
                    cwd = os.getcwd() + f"/tensorboard/{dirs}/"
                    dirs = [dir for dir in os.listdir(cwd) if os.path.isdir(cwd + dir)]


                values = []

                for dir in dirs:

                    summary_iterator = EventAccumulator(cwd + dir).Reload()
                    scalar_tags = summary_iterator.Tags()['scalars']

                    if tag not in scalar_tags:
                        logger.warning(f'{tag} not found in {cwd}{dir}')
                        continue

                    if label == "standard":
                        safety_only = False

                    df = pd.DataFrame.from_records(summary_iterator.Scalars(tag),
                                                   columns=summary_iterator.Scalars(tag)[0]._fields)

                    values.append(df["value"].to_list())
                    #values.append(df["value"][:500].to_list())

                    #if summary_df.empty:

                        #summary_df["episode"] = df["step"] / episode_length



                        #summary_df["value"] = df["value"]/num_runs


                        #if x_min is None:
                        #    x_min = summary_df['episode'].min()
                        #if x_max is None:
                        #    x_max = summary_df['episode'].max()

                    #else:
                        #summary_df["value"] += df["value"] / num_runs


                if values:

                    #values = (-(np.array(values) - 0.5)) + 0.5
                    values = np.array(values)
                    #values = -np.array(values)
                    mean = np.mean(values, axis=0)
                    std_dev = np.std(values, axis=0)

                    if window_size > 1:
                        mean = smooth_data(mean, window_size)
                        std_dev = smooth_data(std_dev, window_size)


                    #TODO: Check PPO Episodes/Steps

                    #["PPO"], ["PPO", "PPO_ZERO"],
            #["PPO", "PPO_SAS", "PPO_LAS"],
            #["PPO", "PPO_EASY", "PPO_LAS"],
            #["PPO", "PPO_EASY_OBS"]

                    # ["MASK_SAS", "MASK", "MASK_INIT"],
                    # ["SHIELD_SAS", "SHIELD", "SHIELD_INIT"],
                    # ["CBF_SAS", "CBF", "CBF_INIT"],
                    # ["MASK_SAS_PUN", "MASK_PUN", "MASK_INIT_PUN"],
                    # ["SHIELD_SAS_PUN", "SHIELD_PUN", "SHIELD_INIT_PUN"],
                    # ["CBF_SAS_PUN", "CBF_PUN", "CBF_INIT_PUN"]

                    if "SAS" in label:
                        color = COLORS_PLOTS[1]
                    elif "INIT" in label:
                        color = COLORS_PLOTS[2]
                    elif label == "PPO_UNTUNED":
                        color= "tab:olive"
                    elif label == "A2C_UNTUNED":
                        color= "tab:cyan"
                    else:
                        color = COLORS_PLOTS[0]

                    # if label == "standard":
                    #     color = COLORS_PLOTS[0]
                    # elif label == "shield":
                    #     color = COLORS_PLOTS[2]
                    # elif label == "mask":
                    #     color = COLORS_PLOTS[1]
                    # elif label == "cbf":
                    #     color = COLORS_PLOTS[3]
                    # else:
                    #     color = COLORS_PLOTS[0]


                   # if color in locals():

                     #   plt.plot(range(1, len(mean)+1), mean, color=color, label=label, linewidth=linewidth)
                     #   plt.fill_between(range(1, len(mean)+1), mean - std_dev, mean + std_dev, color=color, alpha=0.25)

                    #else:
                    #plt.plot(range(1, len(mean) + 1), mean, label=label.replace('_','/'), linewidth=1.25)

                    # plt.gca().axhline(1, linestyle='dotted', color=(102 / 255, 102 / 255, 102 / 255), linewidth=.75,zorder=1)
                    # plt.gca().axhline(0, linestyle='dotted', color='black', linewidth=.75,zorder=1)
                    # plt.gca().axhline(.5, linestyle='dotted', color=(102 / 255, 102 / 255, 102 / 255), linewidth=.5,zorder=1)
                    # plt.gca().axhline(0.25, linestyle='dotted', color=(169 / 255, 169 / 255, 169 / 255), linewidth=.5,
                    #                   zorder=1)
                    # plt.gca().axhline(0.75, linestyle='dotted', color=(169 / 255, 169 / 255, 169 / 255), linewidth=.5,
                    #                   zorder=1)

                    #Safety Measure
                    # plt.gca().axhline(0, linestyle='dotted', color='black', linewidth=.75, zorder=1)
                    # plt.gca().axhline(10, linestyle='dotted', color=(102 / 255, 102 / 255, 102 / 255), linewidth=.5,
                    #                   zorder=1)
                    # plt.gca().axhline(1, linestyle='dotted', color=(102 / 255, 102 / 255, 102 / 255), linewidth=.5,
                    #                   zorder=1)
                    # plt.gca().axhline(5, linestyle='dotted',  color=(102 / 255, 102 / 255, 102 / 255), linewidth=.5,
                    #                   zorder=1)
                    # plt.gca().axhline(0.5, linestyle='dotted',  color=(102 / 255, 102 / 255, 102 / 255), linewidth=.5,
                    #                   zorder=1)

                    plt.gca().axhline(-1, linestyle='dotted', color='magenta', linewidth=.75, zorder=1)
                    plt.gca().axhline(-0.5, linestyle='dotted', color=(102/255,102/255,102/255), linewidth=.5, zorder=1)
                    plt.gca().axhline(-5, linestyle='dotted', color=(169 / 255, 169 / 255, 169 / 255), linewidth=.5,
                                      zorder=1)
                    plt.gca().axhline(-20, linestyle='dotted', color=(169 / 255, 169 / 255, 169 / 255), linewidth=.5,
                                      zorder=1)
                    plt.gca().axhline(-0.25, linestyle='dotted', color=(169 / 255, 169 / 255, 169 / 255), linewidth=.5, zorder=1)
                    plt.gca().axhline(-0.75, linestyle='dotted', color=(169 / 255, 169 / 255, 169 / 255), linewidth=.5, zorder=1)
                    plt.gca().axhline(0, linestyle='dotted', color='black', linewidth=.75, zorder=1)
                    plt.gca().axhline(-10, linestyle='dotted', color=(102/255,102/255,102/255), linewidth=.5, zorder=1)

                    #plt.gca().axhline(0, linestyle='dotted', color='black', linewidth=.75)
                    #plt.gca().axhline(-500, linestyle='dotted', color=(102/255,102/255,102/255), linewidth=.5)
                    #plt.gca().axhline(-1000, linestyle='dotted', color=(102 / 255, 102 / 255, 102 / 255), linewidth=.5)
                    #plt.gca().axhline(-1500, linestyle='dotted', color=(102 / 255, 102 / 255, 102 / 255), linewidth=.5)

                    #plt.plot(range(1, len(mean) + 1), mean, label=label.replace('_', '/'), linewidth=1.25)
                    plt.plot(range(1, len(mean) + 1), mean, color=color, label=label.replace('_', '/'), linewidth=1.25, zorder=2)

                    #plt.plot(range(1, len(mean) + 1), mean, label=label.replace('_','/'), linewidth=linewidth, where=mean>=-1, color="green")
                    plt.fill_between(range(1, len(mean) + 1), mean - std_dev, mean + std_dev, facecolor=color, alpha=0.4, zorder=0)
                    #plt.fill_between(range(1, len(mean) + 1), mean - std_dev, mean + std_dev, alpha=0.25)
                    #plt.fill_between(range(1, len(mean) + 1), -1, mean, alpha=0.4, color="red", where=mean<-1)
                    #plt.fill_between(range(1, len(mean) + 1), -1, mean, alpha=0.4, color="green", where=mean >= -1)

                    #plt.gca().axhline(-1, linestyle='-', color='magenta', linewidth=.75)


                    #if tag == "main/avg_step_reward_rl" or tag == "main/episode_reward" or tag=="main/max_safety_measure":


                    # plt.gca().axhline(0, linestyle='dotted', color='black', linewidth=.75, zorder=1)
                    # plt.gca().set_xlim(right=1000)
                    # plt.gca().set_xticks([0, 250, 500, 750, 1000], minor=False)
                    # minor_ticks = [125, 375, 625, 875]
                    # plt.gca().set_xticks(minor_ticks, minor=True)
                    # plt.gca().yaxis.set_label_position("right")
                    # plt.gca().yaxis.tick_right()

                    # SAFETY
                    # plt.gca().set_ylim(top=.1)
                    # plt.gca().set_ylim(bottom=-1.1)
                    # plt.gca().set_yticks([0, -0.5, -1], minor=False)
                    # plt.gca().set_yticklabels(['$0$', '$-0.5$', '$-1$'])

                    plt.gca().set_yscale("symlog", linthresh=1)
                    plt.gca().set_ylim(top=.25)
                    plt.gca().set_yticks([0, -0.5, -1, -5, -10, -20], minor=False)
                    plt.gca().set_yticklabels(['$0$', '$-0.5$', '$-1$', "$-5$",'$-10$', "$-20$"])
                    plt.gca().set_yticks([-0.25, -0.75, -2 ,-3 ,-4 ,-6, -7, -8, -9, -11, -12,-13,-14,-15,-16,-17,-18,-19], minor=True)
                    plt.gca().set_xlim(right=800)
                    plt.gca().set_xticks([0, 200, 400, 600, 800], minor=False)
                    #plt.gca().set_xticks([0, 250, 500, 750], minor=False)
                    minor_ticks = [100, 300, 500, 700]
                    #minor_ticks = [62.5, 125, 187.5, 312.5, 375, 437.5]
                    plt.gca().set_xticks(minor_ticks, minor=True)
                    #plt.gca().yaxis.set_label_position("right")
                    #plt.gca().yaxis.tick_right()

                    #plt.gca().set_xlim(right=200)
                    #plt.xticks([0,1250,2500])
                    #plt.xticks([0, 1000, 2000], minor=True)

                    # plt.gca().set_yscale("symlog", linthresh=1)
                    # plt.gca().set_xlim(right=800)
                    # #plt.gca().set_ylim(top=1.1)
                    # #plt.gca().set_yticks([0, 0.5, 1], minor=False)
                    # #plt.gca().set_yticks([0.25, 0.75], minor=True)
                    # #plt.gca().set_yticks([0.5], minor=True)
                    # plt.gca().set_ylim(bottom=-0.1)
                    # #plt.gca().set_yticklabels(['$0\%$', '$50\%$', '$100\%$'])
                    # plt.gca().set_xticks([0, 200, 400, 600, 800], minor=False)
                    # plt.gca().set_yticks([10, 5, 1,0.5, 0], minor=False)
                    # plt.gca().set_yticklabels(['$10$',"$5$", '$1$', '$0.5$','$0$'])
                    # plt.gca().set_yticks([9,8,7,6,4,3,2], minor=True)
                    # minor_ticks = [100, 300, 500, 700]
                    # plt.gca().set_xticks(minor_ticks, minor=True)
                    # #plt.gca().yaxis.set_label_position("right")
                    # #plt.gca().yaxis.tick_right()


                    # plt.gca().axhline(20, linestyle='dotted', color=(102 / 255, 102 / 255, 102 / 255), linewidth=.5,
                    #                   zorder=1)
                    # plt.gca().axhline(15, linestyle='dotted', color=(102 / 255, 102 / 255, 102 / 255), linewidth=.5,
                    #                   zorder=1)
                    # plt.gca().axhline(10, linestyle='dotted', color=(102 / 255, 102 / 255, 102 / 255), linewidth=.5,
                    #                   zorder=1)
                    # plt.gca().axhline(5, linestyle='dotted', color=(102 / 255, 102 / 255, 102 / 255), linewidth=.5,
                    #                   zorder=1)
                    # plt.gca().set_xlim(right=800)
                    # plt.gca().set_ylim(bottom=0)
                    # # plt.gca().set_yticklabels(['$0\%$', '$50\%$', '$100\%$'])
                    # plt.gca().set_xticks([0, 200, 400, 600, 800], minor=False)
                    # #plt.gca().set_yticks([10, 5, 1, 0.5, 0], minor=False)
                    # #plt.gca().set_yticklabels(['$10$', "$5$", '$1$', '$0.5$', '$0$'])
                    # #plt.gca().set_yticks([9, 8, 7, 6, 4, 3, 2], minor=True)
                    # minor_ticks = [100, 300, 500, 700]
                    # plt.gca().set_xticks(minor_ticks, minor=True)
                    # plt.gca().yaxis.set_label_position("right")
                    # plt.gca().yaxis.tick_right()

                    # plt.gca().set_ylim(top=250)
                    # plt.gca().set_ylim(bottom=-2000)
                    # plt.gca().set_xlim(right=500)
                    # plt.gca().set_xticks([0, 250, 500], minor=False)
                    # minor_ticks = [62.5, 125, 187.5, 312.5, 375, 437.5]
                    # plt.gca().set_xticks(minor_ticks, minor=True)

                    #plt.gca().set_ylim(top=750)
                    #plt.gca().set_ylim(bottom=-3000)

                    ## TOTAL PLOT
                    # plt.gca().set_xlim(right=800)
                    # #plt.gca().set_yticks([0, -2000, -4000], minor=False)
                    # #plt.gca().set_yticklabels(['$0$', '$-2000$', '$-4000$'])
                    # #minor_ticks = [-1000, -3000, -5000]
                    # #plt.gca().set_yticks(minor_ticks, minor=True)
                    # plt.gca().set_xticks([0, 200, 400, 600, 800], minor=False)
                    # minor_ticks = [100, 300, 500, 700]
                    # plt.gca().set_xticks(minor_ticks, minor=True)
                    # plt.gca().axhline(0, linestyle='dotted', color="black", linewidth=.75, zorder=1)
                    # plt.gca().axhline(-1000, linestyle='dotted', color=(169 / 255, 169 / 255, 169 / 255), linewidth=.5,
                    #                   zorder=1)
                    # plt.gca().axhline(-3000, linestyle='dotted', color=(169 / 255, 169 / 255, 169 / 255), linewidth=.5,
                    #                   zorder=1)
                    # plt.gca().axhline(-5000, linestyle='dotted', color=(169 / 255, 169 / 255, 169 / 255), linewidth=.5,
                    #                   zorder=1)
                    # plt.gca().axhline(-2000, linestyle='dotted', color=(102 / 255, 102 / 255, 102 / 255), linewidth=.5, zorder=1)
                    # plt.gca().axhline(-4000, linestyle='dotted', color=(102 / 255, 102 / 255, 102 / 255), linewidth=.5,
                    #                   zorder=1)



                    #plt.gca().yaxis.set_label_position("right")
                    #plt.gca().yaxis.tick_right()
                    #plt.gca().axhline(-3000, linestyle='dotted', color=(169 / 255, 169 / 255, 169 / 255), linewidth=.5,
                    #                 zorder=1)

        #if x_label == "Episode":
            #TODO: 0/1
            #plt.xlim([0, episodes])
            #assert (x_max).is_integer()
            #print(int(summary_df['episode'].max()))
            #plt.xticks([200 * i for i in range(int(episodes / (200)) + 1)])

        plt.gca().set_xlim(left=0)

        #plt.gca().set_ylim(bottom=-10)
        #plt.gca().set_ylim(top=-1)


        #plt.gca().set_xscale("log")
        #if not safety_only and "main/episode_reward" not in tags: #TODO: Remove if positive reward
        #    try:
        #        plt.gca().set_yscale("log")
        #    except:
        #        print(tags)

        plt.gca().xaxis.grid(True, color='black', linestyle='dotted', linewidth=0.5, alpha=0.6)
        #plt.gca().yaxis.grid(True, color='black', linestyle='dotted', linewidth=0.5, alpha=0.6)

        plt.gca().xaxis.grid(which='minor', color='black', linestyle='dotted', linewidth=0.5, alpha=0.3)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']

        #a2c = Line2D([0], [0], color="tab:olive", label='$\mathrm{PPO}_{\mathrm{untuned}}$') #"$\mathrm{PPO}_{\mathrm{untuned}}$", "$\mathrm{PPO}_{\mathrm{tuned}}$"
        #ppo = Line2D([0], [0], color="tab:blue", label='$\mathrm{PPO}_{\mathrm{tuned}}$')
        #roa = Line2D([0], [0], color="magenta", label='ROA')

        #plt.legend(loc="lower right", handles=[roa])
        #plt.legend(loc="lower right", handles=[a2c,ppo])
        #plt.legend(loc="upper left", fontsize=7, bbox_to_anchor=(1.05, 1))
        #plt.suptitle(tags[0].replace("_",'-'))

        path = None
        if save_as:
            path = f'{os.getcwd()}/{save_as}.pdf'
        finalize_plot(x_label=x_label,
                      y_label=y_label,
                      path=path)





def tf_event_to_plot(dir, tags, x_label='Episode', y_label='', width=5, height=2.5, window_size=1, episode_length=100, save_as=None):
    """ Exports the tf event within ./tensorboard/dir as plotted pdf to ../report/thesis/data/.
    """

    cwd = os.getcwd() + '/tensorboard/'
    if not os.path.isdir(cwd + dir):
        logger.error(f'{cwd + dir} is no valid event file')
        return

    summary_iterator = EventAccumulator(cwd + dir).Reload()
    scalar_tags = summary_iterator.Tags()['scalars']

    if not tags:
        logger.error(f'No tags specified: {scalar_tags}')
        return
    else:
        for tag in tags:
            if tag not in scalar_tags:
                logger.error(f'{tag} not in {scalar_tags}')
                return

    fig = plt.figure()
    setup_plot(fig=fig, width=width, height=height)

    for tag in tags:
        df = pd.DataFrame.from_records(summary_iterator.Scalars(tag), columns=summary_iterator.Scalars(tag)[0]._fields)

        #plt.plot(df['step'] / episode_length, df['value'], 'o',
        #         color='black', markersize=.5)

        if window_size > 1:
            df['value'] = smooth_data(df['value'], window_size)

        if tag == 'main/no_violation':
            plt.fill_between(df['step']/episode_length, 0, df['value'], facecolor='green',alpha=0.5)
            plt.fill_between(df['step'] / episode_length, df['value'], 1, facecolor='red', alpha=0.5)
        else:
            plt.plot(df['step']/episode_length, df['value'], color='magenta')


        if x_label == "Episode":
            #Safety Violation
            plt.xlim([df['step'].min()/episode_length, df['step'].max()/episode_length])
            plt.ylim(0, 1)
            assert (df['step'].max()/episode_length).is_integer()
            plt.xticks([200 * i for i in range(int(df['step'].max()/(200 * episode_length)) + 1)])
            plt.yticks([0, 1])
            plt.gca().set_yticklabels(['$0\%$', '$100\%$'])

        plt.gca().xaxis.grid(True, color='black', linestyle='dotted', linewidth=0.5)
        plt.gca().yaxis.grid(True, color='black', linestyle='dotted', linewidth=0.5)

    path = None
    if save_as:
        path = f'{os.getcwd()}/{save_as}.pdf'
    finalize_plot(x_label=x_label,
                  y_label=y_label,
                  path=path)

def animate(frames, interval=50, dpi=100):
    """
    Returns an animation object given a list of frames.
    Interval adjusts the delay between frames in milliseconds.
    See also: https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
    """

    fig = plt.figure(dpi=dpi)
    im = plt.imshow(frames[0])
    ax = plt.gca()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.close()

    return animation.FuncAnimation(fig, lambda i: im.set_data(frames[i]), frames=len(frames), interval=interval)






if __name__ == '__main__':
    #tf_event_to_plot('60K_A2C_1', 'main/episode_reward')
    pass
    #Moving Average of Safety Violations
    #save_as = 'noSafety2'

    # External legend
    #external_legend(save_as="pdfs/legendROAParted")

    theta_roa = 3.092505268377452
    vertices = np.array([
        [-theta_roa, 12.762720155208534],  # LeftUp
        [theta_roa, -5.890486225480862],  # RightUp
        [theta_roa, -12.762720155208534],  # RightLow
        [-theta_roa, 5.890486225480862]  # LeftLow
    ])

    v1 = np.array([0, 3.436116964863835])
    v2 = np.array([-3.092505268377452, 9.326603190344699])
    #safety_measure_plot(v1, v2, save_as="pdfs/safetyMeasure", vertices=vertices)
    print("Test")

    # gain_matrix = None
    # Gain matrix for current configuration
    #from pendulum.mathematical_pendulum.envs.mathematical_pendulum import MathematicalPendulumEnv
    #print(MathematicalPendulumEnv.gain_matrix())
    gain_matrix = [19.670836678497427,6.351509533724627]

    max_theta = np.pi

    # max_torque = None
    # Torque at [pi/2, 0]
    # from pendulum.mathematical_pendulum.envs.mathematical_pendulum import MathematicalPendulumEnv
    # print(MathematicalPendulumEnv.torque_given_state(gain_matrix, [np.pi/2, 0]))
    max_torque = 30.898877999566082

    max_theta = pi

    #vertices = None
    #boxes = None





    #max_theta = 3.092505268377452
    #theta = fac_2 * (-max_theta)
    #thdot = fac_1 * 3.436116964863835 + fac_2 * 9.326603190344699


    #save_as = "pdfs/euler_plot"
    #safety_measure_plot(v1, v2, K=gain_matrix, vertices=vertices, save_as=save_as)
    #external_legend_res(["$s_0\sim\mathrm{ROA}$", "$s_0=[0\,0]^{\mathsf{T}}$"], colors=["tab:blue", "tab:orange"], save_as="pdfs/tmp", ncols=2,
             #           equi=False)
    #external_legend_res(["$s_0=[0\,0]^{\mathsf{T}}$", "$\mathrm{Default}$", "$|\mathcal{A}|_{\downarrow}$"], colors=["tab:green", "tab:blue", "tab:orange"],
    #                    save_as="pdfs/tmp", ncols=3, equi=False)
    #external_legend_res(["$\mathrm{PPO}_{\mathrm{untuned}}$", "$\mathrm{PPO}_{\mathrm{tuned}}$"], colors=["tab:olive", "tab:blue"], save_as="pdfs/tmp", ncols=2, equi = False)
    #external_legend_res(["A2C", "PPO"],
    #                    colors=["tab:cyan", "tab:olive"], save_as="pdfs/tmp", ncols=2, equi=False)
    #external_legend_res(["Safe", "Unsafe", "ROA"], colors=["green","red", 'magenta'], save_as="pdfs/tmp", ncols=2, equi=False)

    #from thesis.pendulum_roa import PendulumRegionOfAttraction
    #roa = PendulumRegionOfAttraction(vertices=vertices)
    #for v in vertices:
    #    print(v in roa)

    #from thesis.pendulum_roa import PendulumRegionOfAttraction
    #b, v = PendulumRegionOfAttraction.compute_roa()
    #vertices = v
    #boxes = b


    # save_as = "phaseROAParted"
    # phase_plot(K=gain_matrix,
    #            max_torque=max_torque,
    #            max_theta=max_theta,
    #            vertices=vertices,
    #            boxes=boxes,
    #            save_as=save_as)
    # save_as = "phaseSubpavingParted"
    # phase_plot(K=gain_matrix,
    #            max_torque=max_torque,
    #            max_theta=max_theta,
    #            vertices=None,
    #            boxes=boxes,
    #            save_as=save_as)
    # save_as = "phasePolynomParted"
    # phase_plot(K=gain_matrix,
    #            max_torque=max_torque,
    #            max_theta=max_theta,
    #            vertices=vertices,
    #            boxes=None,
    #            save_as=save_as)
    # save_as = "phaseUncontrolled"
    # phase_plot(K=None,
    #            max_torque=None,
    #            max_theta=None,
    #            vertices=None,
    #            boxes=None,
    #            save_as=save_as)
    save_as = "phaseLQR"
    phase_plot(K=gain_matrix,
               max_torque=None,
               max_theta=None,
               vertices=None,
               boxes=None,
               save_as=save_as)
    # save_as = "phaseRestrictedParted"
    # phase_plot(K=gain_matrix,
    #            max_torque=max_torque,
    #            max_theta=max_theta,
    #            vertices=None,
    #            boxes=None,
    #            save_as=save_as)
    #
    # vertices = np.array([
    #     [-theta_roa + 0.5, 12.762720155208534 - 0.5],  # LeftUp
    #     [theta_roa - 0.5, -5.890486225480862 + 0.5],  # RightUp
    #     [theta_roa - 0.5, -12.762720155208534 + 0.5],  # RightLow
    #     [-theta_roa + 0.5, 5.890486225480862 - 0.5]  # LeftLow
    # ])
    #
    # save_as = "phasePolynomReduced"
    # phase_plot(K=gain_matrix,
    #            max_torque=max_torque,
    #            max_theta=max_theta,
    #            vertices=vertices,
    #            boxes=boxes,
    #            save_as=save_as)


