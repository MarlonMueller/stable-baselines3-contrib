"""Utilities for working with plots and tf events."""
from matplotlib.lines import Line2D
from stable_baselines3.common.base_class import BaseAlgorithm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os, shutil, logging
from matplotlib import animation
import matlab
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib import colors as clr
from numpy import pi, sin, cos
import pandas as pd


logger = logging.getLogger(__name__)

COLORS_PLOTS = [
    '#0065bd', #blue
    '#e37222', #orange
    '#008f7c', #green
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

def phase_plot(width=2.5, height=2.5, l=1, m=1, g=9.81, K=None, max_torque=None, max_thdot=None, vertices=None, boxes=None,
               save_as=None):
    """ Generates (and saves) a phase plot of a (controlled) mathematical / simple gravity pendulum."""


    theta, thetadot = np.meshgrid(np.linspace(-1.5 * np.pi, 1.5 * np.pi, 375), np.linspace(-4 * np.pi, 4 * np.pi, 350))

    if K is None:

        start_points = [
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
            [0.8 * pi, -4 * pi]

        ]

        # Controlled system
        if max_torque and max_thdot:

            u = np.dot(np.moveaxis([theta, thetadot], 0, -1), K)
            thetadotdot = g / l * sin(theta) - (1 / (m * l ** 2)) * u

            orange = (abs(np.dot(np.moveaxis([theta, thetadot], 0, -1), K)) <= max_torque)
            orange2 = (abs(thetadot) > max_thdot)#.astype(int)
            orange = -10000*np.logical_and(orange, orange2).astype(int)
            green1 = (abs(np.dot(np.moveaxis([theta, thetadot], 0, -1), K)) <= max_torque)
            green2 = (abs(theta) <= pi) #Artefacts in plot
            green3 = (abs(thetadot) <= max_thdot)
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

        max_thdot = 5.890486225480862 #TODO: E.g. pass as parameter
        b_manual = np.array([[[1.57079633, -max_thdot], [1.96349541, -3.92699082]],
                             [[1.96349541, -max_thdot], [2.35619449, -3.92699082]],
                             [[-1.57079633, 3.92699082], [-1.96349541, max_thdot]],
                             [[-1.96349541, 3.92699082], [-2.35619449, max_thdot]]])
        boxes = np.concatenate((boxes, b_manual), axis=0)

        for b in boxes:
            w, h = b[1] - b[0]

            if abs(b[0,0]) == max_thdot or abs(b[0,1]) == max_thdot:
                print(b[0], b[1])

            if abs(b[1,1]) <= max_thdot and abs(b[0,1]) <= max_thdot:
                box = patches.Rectangle(b[0], w, h,
                                        facecolor='none',
                                        edgecolor='darkturquoise',
                                        linewidth=.85,
                                        linestyle='-',  # {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
                                        alpha=1.0,
                                        zorder=2)
                plt.gca().add_patch(box)

    plt.streamplot(theta, thetadot, thetadot, thetadotdot,
                   density=30,
                   linewidth=linewidth,
                   cmap=cmap,
                   arrowstyle="-|>",
                   arrowsize=0.65,
                   start_points=start_points,
                   color=color,
                   )

    if K is None:
        equilibrium = np.array([[-np.pi, 0, np.pi], [0, 0, 0]])
    else:
        equilibrium = np.array([[0], [0]])

    plt.gca().scatter(equilibrium[0], equilibrium[1], s=8, c=COLORS_HEX['BLUE'], zorder=4)

    plt.yticks([-3* np.pi, -2 * np.pi, -1 * np.pi, 0, 1 * np.pi,2 * np.pi, 3* np.pi])
    plt.gca().set_yticklabels(["$-3\pi$", "$-2\pi$", "$-\pi$", "0", "$\pi$", "$2\pi$", "$3\pi$"])
    plt.xticks([-np.pi, 0, np.pi])
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    plt.xlim([-1.5*pi, 1.5*pi])
    plt.ylim([-3.5*pi, 3.5*pi])
    plt.gca().set_xticklabels(
        ["$-\pi$", "$-\\frac{\pi}{2}$", "0", "$\\frac{\pi}{2}$", "$\pi$"])

    plt.gca().xaxis.grid(True, linestyle='dotted', linewidth=0.5)
    plt.gca().yaxis.grid(True, linestyle='dotted', linewidth=0.5)
    #plt.gca().axhline(0, color='gray', linewidth=0.5)

    if save_as:
        finalize_plot(x_label='$\\theta[\mathrm{rad}]$',
                      y_label='$\dot{\\theta}[\\frac{\mathrm{rad}}{\mathrm{s}}]$',
                      path=f'{os.getcwd()}/../../report/thesis/figures/{save_as}.pdf')
    else:
        finalize_plot(x_label='$\\theta[\mathrm{rad}]$',
                      y_label='$\dot{\\theta}[\\frac{\mathrm{rad}}{\mathrm{s}}]$')


def external_legend(save_as=None, width=1., height=.25):

    fig = plt.figure()
    setup_plot(fig=fig, width=width, height=height)
    plt.gca().axis('off')

    b_line = Line2D([0], [0], color="white", markerfacecolor=COLORS_HEX['BLUE'], marker='o', label='Equilibrium')
    g_line = Line2D([0], [0], color='green', label='$\\tau\\leq\\tau_{\mathrm{max}}\;\mathrm{and}\;\dot{\\theta}\\leq\dot{\\theta}_{\mathrm{max}}$')
    o_line = Line2D([0], [0], color='orange', label='$\\tau\\leq\\tau_{\mathrm{max}}\;\mathrm{and}\;\dot{\\theta}>\dot{\\theta}_{\mathrm{max}}$')
    r_line = Line2D([0], [0], color='red', label='$\\tau >\\tau_{\mathrm{max}}$')
    t_line = Line2D([0], [0], color='darkturquoise', label='ROA (Subpaving)')
    m_line = Line2D([0], [0], color='magenta', label='ROA (Polygon)')

    plt.gca().legend(handles=[g_line, o_line, r_line, b_line, t_line, m_line], frameon=True, loc='center', ncol=2)

    if save_as:
        plt.savefig(path = f'{os.getcwd()}/{save_as}.pdf', dpi=1000, bbox_inches='tight')
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

                    values = np.array(values)

                    mean = np.mean(values, axis=0)
                    std_dev = np.std(values, axis=0)

                    if window_size > 1:
                        mean = smooth_data(mean, window_size)

                    #TODO: Check PPO Episodes/Steps

                    linewidth = 1

                    if label == "standard":
                        color = COLORS_PLOTS[0]
                    elif label == "shield":
                        color = COLORS_PLOTS[2]
                    elif label == "mask":
                        color = COLORS_PLOTS[1]
                    elif label == "cbf":
                        color = COLORS_PLOTS[3]
                    else:
                        color = COLORS_PLOTS[0]

                   # if color in locals():

                     #   plt.plot(range(1, len(mean)+1), mean, color=color, label=label, linewidth=linewidth)
                     #   plt.fill_between(range(1, len(mean)+1), mean - std_dev, mean + std_dev, color=color, alpha=0.25)

                    #else:
                    plt.plot(range(1, len(mean) + 1), mean, label=label.replace('_','/'), linewidth=linewidth)
                    plt.fill_between(range(1, len(mean) + 1), mean - std_dev, mean + std_dev, alpha=0.25)




        #if x_label == "Episode":
            #TODO: 0/1
            #plt.xlim([0, episodes])
            #assert (x_max).is_integer()
            #print(int(summary_df['episode'].max()))
            #plt.xticks([200 * i for i in range(int(episodes / (200)) + 1)])

        #plt.gca().set_ylim(bottom=-1)
        #plt.gca().set_ylim(top=-1)

        #plt.gca().set_xscale("log")
        #if not safety_only and "main/episode_reward" not in tags: #TODO: Remove if positive reward
        #    try:
        #        plt.gca().set_yscale("log")
        #    except:
        #        print(tags)

        plt.gca().xaxis.grid(True, color='black', linestyle='dotted', linewidth=0.5)
        plt.gca().yaxis.grid(True, color='black', linestyle='dotted', linewidth=0.5)

        plt.legend(loc="upper left", fontsize=7, bbox_to_anchor=(1.05, 1))

        plt.suptitle(tags[0].replace("_",'-'))

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
    # Avg. masked out actions
    # Avg. shield use
    # % of max force used or more?
    # max distance middle
    # max distance border
    # speed / reward
    # max theta/theta dot used / avg

    #max_thdot = 5.890486225480862
    #vertices = np.array([
    #    [-pi, max_thdot],  # LeftUp
    #    [-0.785398163397448, max_thdot],  # RightUp
    #    [pi, -max_thdot],  # RightLow
    #    [0.785398163397448, -max_thdot]  # LeftLow
    #])

    gain_matrix = [19.670836678497427,6.351509533724627]
    print(torque_given_state(gain_matrix=gain_matrix, state=[pi/2, 0]))


    #tf_event_to_plot('80K_A2C_1', ['main/no_violation'],#, 'main/shield_activations'],
    #                 y_label='', save_as=save_as,
    #                 window_size=51)
    # tf_event_to_plot('DEBUG_E_1', 'main/theta')

