import base64
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage import img_as_ubyte
from skimage import io
from skimage.feature import plot_matches
import streamlit as st

FIGSIZE = (15, 10)
COLUMNS = 3
ROWS = 2
SHOW_PLOTS = True


def show_plot():
    if SHOW_PLOTS:
        st.pyplot()


def save_plot(filename):
    plt.savefig(filename)



def plot_inliers(src, dest, src_keypoints, dest_keypoints, matches):
    plt.close()
    plt.figure(figsize=FIGSIZE)
    ax = plt.axes()
    ax.axis("off")
    ax.set_title(f"Inlier correspondences: {len(matches)} points matched")
    plot_matches(ax, src, dest, src_keypoints, dest_keypoints,
                 matches)
    plt.tight_layout()
    show_plot()


def plot_warps(corners, output_shape=None, min_coords=None, max_coords=None, img=None):
    plt.close()
    np.random.seed(0)
    plt.figure(figsize=(15, 5))
    ax = plt.axes()

    for coords in corners:
        ax.add_patch(Polygon(coords, closed=True, fill=False, color=np.random.rand(3)))

    if max_coords is not None:
        plt.xlim(min_coords[0], max_coords[0])
        plt.ylim(max_coords[1], min_coords[1])

    if output_shape is not None:
        plt.xlim(0, output_shape[1])
        plt.ylim(output_shape[0], 0)

    if img is not None:
        plt.imshow(img)
    plt.axis('off')
    plt.title('Border visualization')
    plt.tight_layout()
    show_plot()





def plot_result( result):
    plt.close()
    plt.figure(figsize=FIGSIZE)
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    show_plot()



