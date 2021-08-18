"""Utilities for working with plots and tf events."""

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os, shutil, logging
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

#TODO: Refactor!

# if not env_util.is_wrapped(self.env, SafetyWrapper):
            #raise RuntimeError("Environment is not wrapped with a ``SafetyWrapper``")

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

def setup_plot(fig, width, height, font_size=11):
    """ Used to initialize uniform plots.

    Parameters
    ----------
    fig
    width
    height
    font_size

    """

    fig.set_size_inches(width, height)

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = font_size



    # plt.gca().tick_params(axis="x", direction="in")
    # plt.gca().tick_params(axis="y", direction="in")


def finalize_plot(fig=None, width=.0, height=.0, x_label='', y_label='', path=None):
    """ Finalize and save plots.

    Parameters
    ----------
    fig
    width
    height
    x_label
    y_label
    path

    """

    # plt.grid

    plt.gca().tick_params(axis="x", direction="out", zorder=10)  # =0)
    plt.gca().tick_params(axis="y", direction="out", zorder=10)  # length=0)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if path:
        plt.savefig(path, dpi=1000, bbox_inches='tight')
    plt.show()


def remove_tf_logs(*dirs):
    """ Removes directories within ./tensorboard.

    Parameters
    ----------
    dirs
        Names of the directories within ./tensorboard to be removed.
        If no dirs are given, all directories are renamed.

    """

    cwd = os.getcwd() + '/tensorboard/'

    if not dirs:
        dirs = os.listdir(cwd)

    for dir in dirs:
        shutil.rmtree(cwd + dir, ignore_errors=True)


def rename_tf_events(*dirs):
    """ Renames the tf events within ./tensorboard/*dirs to tfevents.event.

    Notes
    ----------
    One directory only contains one tf event.

    Parameters
    ----------
    dirs
        Names of the directories within ./tensorboard containing the tf events.
        If no dirs are given, all events within all dirs are renamed.

    """

    cwd = os.getcwd() + '/tensorboard/'

    if not dirs:
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


def tf_event_to_plot(dir, tags, x_label='Episode', y_label='', width=5, height=2.5, window_size=1, episode_length=100, save_as=None):
    """ Exports the tf event within ./tensorboard/dir as plotted pdf to ../report/thesis/data/.

    Parameters
    ----------
    dir
        Names of the directories within ./tensorboard containing the tf event.
    tags
        Variable amount of tags.
        For every given tag a new .pdf file is generated in ../report/thesis/data/.
    x_label
    y_label
    width
    height
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
            event_tags = scalar_tags
            if tag not in event_tags:
                logger.error(f'{tag} not in {event_tags}')
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
            #plt.ylim([df['value'].min(), df['value'].max()])
            plt.ylim(0, 1)
            #plt.ylim(-.25, 1.25)

            assert (df['step'].max()/episode_length).is_integer()

            plt.xticks([100 * i for i in range(int(df['step'].max()/(100 * episode_length)) + 1)])
            plt.yticks([0, 1])
            plt.gca().set_yticklabels(['$0\%$', '$100\%$'])

        #plt.gca().xaxis.grid(True)
        plt.gca().xaxis.grid(True, color='black', linestyle='dotted', linewidth=0.5)
        plt.gca().yaxis.grid(True, color='black', linestyle='dotted', linewidth=0.5)
        #plt.gca().yaxis.grid(True)

    path = None
    if save_as:
        path = f'{os.getcwd()}/../report/thesis/figures/{save_as}.pdf'
    finalize_plot(x_label=x_label,
                  y_label=y_label,
                  path=path)

        # Export as csv
        # file = f'{dir}_{tag}.dat'.replace('/', '_')
        # df.to_csv(f'{os.getcwd()}/../report/thesis/data/{file}', sep='\t', encoding='utf-8', index=False)


if __name__ == '__main__':
    #tf_event_to_plot('60K_A2C_1', 'main/episode_reward')
    #Moving Average of Safety Violations
    save_as = 'noSafety2'
    # Avg. masked out actions
    # Avg. shield use
    # % of max force used or more?
    # max distance middle
    # max distance border
    # speed / reward
    # max theta/theta dot used / avg

    tf_event_to_plot('80K_A2C_1', ['main/no_violation'],#, 'main/shield_activations'],
                     y_label='', save_as=save_as,
                     window_size=51)
    # tf_event_to_plot('DEBUG_E_1', 'main/theta')
