# Created by @svilella on 18/04/2022

import matplotlib.pyplot as plt
from matplotlib.colors import DivergingNorm
from math import log, e
import numpy as np
import matplotlib as mpl


def entropy(labels, base=None):
    """ Computes entropy of label distribution. """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent


def scatter(df, middle, cmap, size_par, top_y_val, xy=(1.67, 0.51), figsize=(16, 6), top_y=False, save=False,
            path=None):
    # xy = (1.67, 0.51)

    """
    :param df: pandas dataframe to plot. Has fixed structure (function is hardcoded to specific work)
    :param middle: center of the cmap
    :param cmap: cmap
    :param size_par: scale factor to resize dots
    :param top_y_val: max of y axis, check param top_y
    :param xy: position of colorbar title
    :param figsize: figsize, default (16, 6)
    :param top_y: bool. if True, sets [0, top_y_val] as interval for y axis
    :param save: bool. if True, saves the fig to path
    :param path: str, path where to save fig.
    :return: _scatter, scatterplot object
    """

    fig, ax = plt.subplots(figsize=figsize)

    size = df['users'] * size_par  # adjust dot size
    _scatter = plt.scatter(df['entropy'], df['avg_botscore'], s=size, linewidths=0.1, edgecolors='k',
                           norm=DivergingNorm(middle), c=df['avg_u'], cmap=cmap)
    plt.xlabel('Entropy', fontsize=18)
    plt.ylabel('Average BotScore', fontsize=18)
    ax.set_ylim(bottom=0)

    if top_y:
        ax.set_ylim(bottom=0, top=top_y_val)

    # define legend object and put it in handles
    handles = _scatter.legend_elements(prop="sizes", alpha=0.6, num=6)

    # adjust legend labels
    plt.legend(handles=handles[0], labels=['$\\mathdefault{' + str(i * 250) + '}$' for i in range(1, 7, 1)],
               prop={'size': 14}, title='Shares')

    # colorbar
    ticks = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    mappable = mpl.cm.ScalarMappable(norm=DivergingNorm(middle, vmin=0, vmax=0.05), cmap=cmap)
    fig.colorbar(mappable, shrink=.5, aspect=10, pad=.03, ticks=ticks)

    # colorbar title as annotation (there is no title param in colorbar :| )
    plt.annotate(s='Average U', xy=xy, annotation_clip=False, size=14)

    if save:
        plt.savefig(path, dpi=300, transparent=True)

    return _scatter


cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

cmaps += ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
          'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
          'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

cmaps += ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
          'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
          'hot', 'afmhot', 'gist_heat', 'copper']

cmaps = [c + '_r' for c in cmaps]

BEST_COLORMAP = 'RdBu_r'


def eval_item(x, a, b):
    """
    Use this to categorise an item based on its value.
    :param x: item to categorise
    :param a: lower bound
    :param b: upper bound
    :return: y (str), label for x
    """
    if x < a:
        y = 'low'
    elif a < x < b:
        y = 'medium'
    else:
        y = 'high'

    return y


def u_index(inputdict, subpart, rnd_subpart):
    """
    :param inputdict: dict, contains information about the number of (un)reliable content per user.
    Required form is {'user_id':{'num_rel': #, 'num_unrel': #, 'lost': #}}
    :param subpart: dict, community structure. Required form is {'user_id':'community'}
    :param rnd_subpart: dict, randomised community structure. Required form is {'user_id':'community'}
    :return: new_score, pandas dataframe. Columns:
        user_id: twitter user id
        top5community: id of the community
        num_rel: number of tweets w/ content from reliable sources
        num_unrel: number of tweets w/ content from unreliable sources
        num_lost: number of tweets w/o content from neither reliable nor unreliable sources
        tweets: num_rel+num_unrel, concur to the calculation of U
        tweets_norm: n_tweets normalised over max(n_tweets)
        linear_fs: disregard
        harm_fs: Untrustworthiness index of the user
    """

    import pandas as pd
    from scipy.stats import hmean

    new_score = pd.DataFrame()
    new_score['user_id'] = [str(x) for x in inputdict.keys() if x in subpart.keys()]
    new_score['user_id'] = new_score['user_id'].astype(str)
    new_score['top5community'] = new_score['user_id'].map(subpart)
    if rnd_subpart:
        assert isinstance(rnd_subpart, dict)
        new_score['rnd_community'] = new_score['user_id'].map(rnd_subpart)
    new_score['num_rel'] = [inputdict[x]['num_rel'] for x in inputdict.keys() if x in subpart.keys()]
    new_score['num_unrel'] = [inputdict[x]['num_unrel'] for x in inputdict.keys() if x in subpart.keys()]
    new_score['num_lost'] = [inputdict[x]['lost'] for x in inputdict.keys() if x in subpart.keys()]

    new_score['tweets'] = new_score['num_rel'] + new_score['num_unrel']
    new_score['tweets_norm'] = new_score['tweets'] / new_score['tweets'].max()

    new_score['linear_fs'] = new_score['num_unrel'] / (new_score['num_unrel'] + new_score['num_rel'])  # old method

    new_score = new_score.fillna(0)

    norm_tw = new_score['tweets_norm'].to_list()
    fake = new_score['linear_fs'].to_list()

    eps = 0.0001
    new_score['harm_fs'] = [hmean([a + eps, b + eps]) for a, b in zip(norm_tw, fake)]
    new_score['norm_harm_fs'] = new_score['harm_fs'] / new_score['harm_fs'].max()  # Untrustworthiness index

    return new_score


def get_screenname(keys, ids):
    """

    :param keys: path to API keys file.
    :param ids: iterable, contains user ids for which we need screen names.
    :return: names (list), exception (list).

    Names follows the same order as ids. If a user is not found, name will be 'not_found'. It will be listed in
    exceptions as (index, id).
    """
    import tweepy
    from tqdm import tqdm

    with open(keys) as keys_file:
        lines = keys_file.readlines()
        consumer_key = str(lines[0].rstrip().split()[0])
        consumer_secret = str(lines[1].rstrip().split()[0])
        access_token = str(lines[2].rstrip().split()[0])
        access_token_secret = str(lines[3].rstrip().split()[0])

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    names = []
    exception = []

    for i, x in tqdm(enumerate(ids)):
        try:
            names.append(api.get_user(x).screen_name)
        except Exception as _e:
            exception.append((i, x, _e))
            names.append('not_found')

    return names, exception


def clean_urls(picklefile, df_urls, save=False):
    """
    :param picklefile: pickle of retweeted urls chains
    :param df_urls: pd.DataFrame of urls for which we have entropy info, with 'url' column
    :param save: bool. If True, dumps a urls_clean.dict pickle file.
    :return: urls_clean, dict.
    """
    import pickle

    # urls che vengono retwittati
    with open(picklefile, 'rb') as fi:
        temp_urls_rt = pickle.load(fi)

    # {'url':[shares list]} where each
    # item of shares list is formatted as [id_us, id_rt, id_us_orig, id_tw, timestamp, timestamp_orig]

    urls_rt = {}
    for k, v in temp_urls_rt.items():
        flat_list = [item for sublist in v for item in sublist]
        urls_rt[k] = flat_list
    del temp_urls_rt

    # urls for which we have entropy info
    urls_clean = {k: v for k, v in urls_rt.items() if k in set(df_urls.url)}

    if save:
        with open('outputs/urls_clean.dict', 'wb') as fo:
            pickle.dump(urls_clean, fo)

    return urls_clean


def filter_df(url_dict, df_urls, untrust_dict, bs_results_dict):
    import pandas as pd

    df = pd.DataFrame()

    df['url'] = [x for x in list(url_dict.keys())]
    df['users'] = [len(v) for v in url_dict.values()]
    df['unique_users'] = [len(set([i[0] for i in v])) for v in url_dict.values()]
    print(len(df))
    assert (len(df.dropna()) != 0)
    df = df.merge(df_urls[['url', 'entropy']], on='url')
    print(len(df))
    assert (len(df.dropna()) != 0)

    sub = [[i[0] for i in v] for v in url_dict.values()]
    means = []

    for i in sub:
        vals = [untrust_dict[a] for a in i if a in untrust_dict.keys()]
        means.append(np.mean(vals))
    assert (len(means) > 0)

    overall_scores_dict = {k: bs_results_dict[k]['raw_scores']['universal']['overall'] for k in bs_results_dict.keys()}

    botmeans = []
    for i in sub:
        botvals = [overall_scores_dict[a] for a in i if a in overall_scores_dict.keys()]
        botmeans.append(np.mean(botvals))
    assert (len(botmeans) > 0)

    df['avg_u'] = means
    df['avg_botscore'] = botmeans

    print(len(df))
    # assert(len(df.dropna() > 0)

    df = df.dropna()
    print(len(df))

    return df


def mainplot(ax, posterior_by_class, colors, xlabel, ylabel, font):
    import matplotlib.font_manager as fm
    fm.fontManager.ttflist += fm.createFontList(['resources/josefin_sans_regular.ttf'])
    fm.fontManager.ttflist += fm.createFontList(['resources/Roboto-Regular.ttf'])
    # FONT = 'Roboto'

    for c in ['low', 'medium', 'high'][::-1]:
        x = posterior_by_class[c][0]
        y = posterior_by_class[c][1]

        if c == 'high':
            size, zorder, za = 3, 5, 1
        elif c == 'medium':
            size, zorder, za = 2.5, 6, 2
        else:
            size, zorder, za = 2, 7, 3

        plt.plot(x, y, label=c, color=colors[c], linewidth=size, zorder=zorder)
        plt.plot(x, y, color='white', linewidth=size + 4, alpha=1, zorder=zorder)
        plt.plot(x, y, color=colors[c], linewidth=size, zorder=zorder)
        plt.fill_between(x, 0, y, color='white', alpha=1, zorder=1)
        plt.fill_between(x, 0, y, color=colors[c], alpha=0.5, zorder=za)

        ax.scatter([x[-1]], [y[-1]], s=150, color=colors[c], zorder=9)
        ax.scatter([x[-1]], [y[-1]], s=120, color='white', zorder=10)
        ax.scatter([x[-1]], [y[-1]], s=80, color=colors[c], zorder=11)

    ax.set_xlabel(xlabel, fontsize=14, fontfamily=font)
    ax.set_ylabel(ylabel, fontsize=14, fontfamily=font)


#     ax.legend()

def hist_plot(ax, values, color, label):
    ax.hist(values, bins=10, histtype='stepfilled', color=color, label=label)
    ax.set_ylim((0, 180))
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])

    if label == 'medium':
        shift = .0

    elif label == 'low':
        shift = .045
    else:
        shift = .033

    legend = plt.legend(loc='upper right', frameon=False, bbox_to_anchor=(0.99 - shift, 0.8))

    if label == 'high':
        ax.spines['top'].set_visible(False)
    elif label == 'medium':
        ax.set_ylabel("Distribution of U \nby entropy class", labelpad=20)


def posterior_by_class(df, col, rt_thr):
    pbc = {}

    for eclass in df.entropy_class.unique():
        posterior = []
        temp_df = df[df['entropy_class'] == eclass]
        b_thresholds = np.linspace(0, temp_df[col].max(), 100)
        for b_thr in b_thresholds:
            try:
                a = len(temp_df[(temp_df['count'] > rt_thr) & (temp_df[col] >= b_thr)])
                b = len(temp_df[temp_df['count'] > rt_thr])
                likelihood = a / b

                c = len(temp_df)
                prior = b / c

                d = len(temp_df[temp_df[col] >= b_thr])
                marginal = d / c

                post = (likelihood * prior) / marginal
                posterior.append(post)
                pbc[eclass] = [b_thresholds, posterior]

            except:
                print('error! b_thr: {}, d: {}, entropy class: {}'.format(b_thr, d, eclass))

    return pbc
