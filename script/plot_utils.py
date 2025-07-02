import os
import numpy as np
import pandas as pd
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn import metrics
from scipy.stats import gaussian_kde, spearmanr

def compute_metrics(df, x, y, not_print=True):
    df_x, df_y = df[x].tolist(), df[y].tolist()
    mse_test = metrics.mean_squared_error(df_x, df_y)
    rmse_test = np.sqrt(mse_test)
    mae_test = metrics.mean_absolute_error(df_x, df_y)
    r2_test = metrics.r2_score(df_x, df_y)
    if not not_print:
        print("MSE:", mse_test)
        print("RMSE:", rmse_test)
        print("MAE:", mae_test)
        print("R-squared:", r2_test)
    return mse_test, rmse_test, mae_test, r2_test

def plot_scatter(df, x, y, save_as=None, figsize=(6,6), dpi=300):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    mse_test, rmse_test, mae_test, r2_test = compute_metrics(df, x, y, not_print=False)
    pearsonR = df[[x, y]].corr('pearson').iloc[0, 1]
    spearmanR = df[[x, y]].corr('spearman').iloc[0, 1]
    spearmanR, spearmanP = spearmanr(df[x], df[y])

    g = sns.JointGrid(data=df, x=x, y=y, height=10)

    xy = df[[x, y]].values.T
    density = gaussian_kde(xy)(xy)
    density = (density - density.min()) / (density.max() - density.min())

    palette = sns.color_palette("Blues_r", as_cmap=True)
    g.plot_joint(sns.scatterplot, alpha=1, size=1, legend=False)

    ax = g.ax_joint
    sns.regplot(data=df, x=x, y=y, scatter=False, ax=ax, color='#1F4788', label='Regression Line')

    g.plot_marginals(sns.kdeplot, fill=True, common_norm=True, alpha=0.5, bw_adjust=0.2)
    
    text = f'$R^2$={r2_test:.4f}\n$Spearman$ $R$={spearmanR:.4f}\n'
    if spearmanP<0.0001:
        text+=f'$p$<0.0001'
    else:
        text+=f'$p$={spearmanP:.4f}'
    
    ax.text(0.95, 0.05, text, 
            transform=ax.transAxes, fontsize=25,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    ax.plot([df[x].min(), df[x].max()], [df[x].min(), df[x].max()], c="black", alpha=0.5, linestyle='--', label='x=y')
    ax.legend(fontsize=25, loc='upper left')
    ax.tick_params(labelsize=15)
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)

    fig.tight_layout()
    if save_as:
        os.makedirs(os.path.split(save_as)[0], exist_ok=True)
        plt.savefig(save_as, dpi=dpi, bbox_inches='tight', format=save_as.split('.')[-1])
    plt.show()


def plot_violin(df):
    plt.figure(figsize=(3, 5), dpi=300)
    violin = sns.violinplot(x='Group', y='Value', data=df, palette='Blues', inner='quart')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=10)
    plt.xlabel('')
    plt.ylabel('NGS-validated Fold Change\nin Viability, Norm. to AAV2.WT', fontsize=10)
    plt.axhline(1, c='green', linestyle='--')
    custom_legend = [
        Line2D([], [], color='green', linewidth=2, linestyle='--', label='WT'),
    ]
    plt.legend(handles=custom_legend, fontsize=10)
    plt.ylim(0)
    plt.xticks(rotation = 45, fontsize=10)

    plt.tight_layout()
    plt.show()
