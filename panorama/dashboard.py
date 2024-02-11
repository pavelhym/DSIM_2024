#import pandas as pd
import matplotlib.pyplot as plt
import datetime
import random
from datetime import timedelta
from streamlit_card import card
from streamlit_player import st_player
from st_on_hover_tabs import on_hover_tabs
from streamlit_extras.metric_cards import style_metric_cards
import altair as alt
import numpy as np
import streamlit as st
import yaml
from yaml.loader import SafeLoader
from streamlit_elements import elements, mui, html, dashboard, editor, lazy, sync
from streamlit_option_menu import option_menu
import requests
import io
from streamlit_extras.switch_page_button import switch_page 
from PIL import Image

from src import panorama 
from src import plots 

import os
import shutil
import traceback

from skimage import io
import numpy as np
st.set_option('deprecation.showPyplotGlobalUse', False)



from src.functions import  multiselect_with_all, plotly_noedge
st.set_page_config(layout="wide", page_icon='src/bicocca.svg', page_title='Digita signal')
#st.markdown('<style>' + open('./src/style.css').read() + '</style>', unsafe_allow_html=True)




#disable fullscreen
hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
'''

st.markdown(hide_img_fs, unsafe_allow_html=True)

def delete_files(directory_path):
    # Get the list of files in the directory
    files = os.listdir(directory_path)

    # Iterate through the files and delete each one
    for file in files:
        file_path = os.path.join(directory_path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")




#change tab fontsize
css_tabs = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:2rem;
    }
</style>
'''

st.markdown(css_tabs, unsafe_allow_html=True)

#change fontsize
css = '''
<style>
    .st-emotion-cache-q8sbsg p {
        font-size: 25px;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

#change font
css = '''
<style>
    .st-emotion-cache-q8sbsg {
        font-family: Pragmatica;
    }
</style>
'''
#change sidebar color
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #111;
    }
</style>
""", unsafe_allow_html=True)
st.markdown(css, unsafe_allow_html=True)

#hide made by streamlit
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 





# 1. as sidebar menu
with st.sidebar:
    selected = on_hover_tabs(tabName=["Welcome page", "Image warp"], 
                        iconName=['info', 'palette'], default_choice=0)



if (selected == 'Welcome page'):
    st.markdown('<p style="font-family: Pragmatica; font-size: 75px;">Digital signal Project</p>', unsafe_allow_html=True)

    st.markdown(
    """
    ## It is our project:
    **Image wrap**

    """
    )
    







elif selected == 'Image warp':
    st.markdown('<p style="font-family: Pragmatica; font-size: 75px;">Image warp online</p>', unsafe_allow_html=True)


    if "camera_dis" not in st.session_state:
        st.session_state['camera_dis'] = False


    upload_option = st.selectbox(
   "How would you like upload photos?",
   ("Files", "Camera"),
   index=1)

    if upload_option == "Camera":

        picture = st.camera_input("Take a picture", disabled=st.session_state['camera_dis'])

        if "image_counter" not in st.session_state:
            st.session_state['image_counter'] = 0

    
        if picture:
            with open(os.path.join("tempDir",f"{st.session_state['image_counter']}.jpg"),"wb") as f: 
                f.write(picture.getbuffer())         
            st.session_state['image_counter'] += 1



    elif upload_option == "Files" :
        uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True)
        for picture in uploaded_files:
            with open(os.path.join("tempDir",picture.name),"wb") as f: 
                f.write(picture.getbuffer())         
            



    
    if st.button('Lets warp!'):
        st.session_state['camera_dis'] = True
        # ------------------------------------------------------------------------------------------
        # Part 0
        # ------------------------------------------------------------------------------------------

        pano_image_collection = io.ImageCollection(f'tempDir/*',
                                                    load_func=lambda f: io.imread(f).astype(np.float64) / 255)


        # ------------------------------------------------------------------------------------------
        # Part 1
        # ------------------------------------------------------------------------------------------

        img = pano_image_collection[0]
        keypoints, descriptors = panorama.find_orb(img)


        # ------------------------------------------------------------------------------------------
        # Part 2 and 3
        # ------------------------------------------------------------------------------------------

        src, dest = pano_image_collection[0], pano_image_collection[1]
        src_keypoints, src_descriptors = panorama.find_orb(src)
        dest_keypoints, dest_descriptors = panorama.find_orb(dest)

        robust_transform, matches = panorama.ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, return_matches=True)

        plots.plot_inliers(src, dest, src_keypoints, dest_keypoints, matches)

        # ------------------------------------------------------------------------------------------
        # Part 4
        # ------------------------------------------------------------------------------------------

        keypoints, descriptors = zip(*(panorama.find_orb(img) for img in pano_image_collection))
        forward_transforms = tuple(panorama.ransac_transform(src_kp, src_desc, dest_kp, dest_desc)
                                    for src_kp, src_desc, dest_kp, dest_desc
                                    in zip(keypoints[:-1], descriptors[:-1], keypoints[1:], descriptors[1:]))





        simple_center_warps = panorama.find_simple_center_warps(forward_transforms)
        corners = np.flip(tuple(panorama.get_corners(pano_image_collection, simple_center_warps)), axis= None)
        min_coords, max_coords = panorama.get_min_max_coords(corners)
        center_img = pano_image_collection[(len(pano_image_collection) - 1) // 2]

        plots.plot_warps( corners, min_coords=min_coords, max_coords=max_coords, img=center_img)

        final_center_warps, output_shape = panorama.get_final_center_warps(pano_image_collection, simple_center_warps)
        corners = np.flip(tuple(panorama.get_corners(pano_image_collection, final_center_warps)), axis= None)


        # ------------------------------------------------------------------------------------------
        # Part 5
        # ------------------------------------------------------------------------------------------

        result = panorama.merge_pano(pano_image_collection, final_center_warps, output_shape)

        plots.plot_result( result)

        # ------------------------------------------------------------------------------------------
        # Part 6
        # ------------------------------------------------------------------------------------------

        img = pano_image_collection[0]

        result = panorama.gaussian_merge_pano(pano_image_collection, final_center_warps, output_shape)

        plots.plot_result(result)
        directory_path = 'tempDir'
        delete_files(directory_path)

