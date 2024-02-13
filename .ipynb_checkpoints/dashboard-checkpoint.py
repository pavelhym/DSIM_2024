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
import tempfile

from panorama.src import panoramafunc 
from panorama.src import plots 
import librosa
import keras
from streamlit_mic_recorder import mic_recorder
from voice.audiofunc import predict
from voice.audiofunc import detect

from gan.ganfunc import show_gan_imgs

import os
import shutil
import traceback

import time
 


from skimage import io
import numpy as np
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(layout="wide", page_icon='panorama/src/bicocca.svg', page_title='Digita signal')
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
    selected = on_hover_tabs(tabName=["Welcome page","Voice emotion", "Image warp", "Image generation"], 
                        iconName=['info','record_voice_over',  'palette', 'smart_toy'], default_choice=0)



if (selected == 'Welcome page'):
    st.markdown('<p style="font-family: Pragmatica; font-size: 75px;">DSIM 2024 Project</p>', unsafe_allow_html=True)



    st.markdown(
            """
            ## Project's parts:
            **Voice emotion**
            - Detect the emotion of your voice
            
            **Image wrap**
            - Create panorama by yourself and look through steps
            - Choose common directory to uplad images from any device
            - Choose private directory to use only photos from this device
    
            **Image generation**
            - Generate image of car by trained model
            

            ***
            ***


    
            Done by <a href="https://www.linkedin.com/in/pavel-shumkovskii-b9bb3a233/">Pavel Shumkovskii</a> and Cristian Ceccarelli
            
            """,unsafe_allow_html=True
            )


    

elif selected == 'Voice emotion':
    st.markdown('<p style="font-family: Pragmatica; font-size: 55px;">Voice emotion detection</p>', unsafe_allow_html=True)


    temp_dir_aud_obj = tempfile.TemporaryDirectory()
    temp_dir_aud = temp_dir_aud_obj.name



    upload_audio = st.selectbox(
   "How would you like to upload photos?",
   ("Files", "Microphone"),
   index=1)

    if upload_audio == 'Microphone':
    
        st.write("Record your voice, play the recorded audio and predict the emotion:")
    
        audio=mic_recorder(key='recorder')
        if audio:
          st.audio(audio['bytes'])
          if st.button('Predict Emotion'):
            audio_array = np.frombuffer(audio['bytes'], dtype=np.int16)
            result_pano = predict(audio=audio_array, sr=audio['sample_rate'])
            st.write(f"You are {detect(result_pano)}!")

    elif upload_audio == 'Files':
        st.write("Upload audio and predict the emotion:")
        audio = st.file_uploader("Upload audio", accept_multiple_files=False, type = ["wav", "mp3"])
        if audio is not None:
            st.audio(audio)
            with open(os.path.join(temp_dir_aud,audio.name),"wb") as f: 
                f.write(audio.getbuffer())         
        

        if st.button('Predict Emotion'):
            audio, sr = librosa.load(f'{temp_dir_aud}/{audio.name}')
            result_pano = predict(audio=audio, sr=sr)
            st.subheader(f"You are {detect(result_pano)}!")


elif selected == 'Image warp':
    st.markdown('<p style="font-family: Pragmatica; font-size: 55px;">Image warp online</p>', unsafe_allow_html=True)
    #with tempfile.TemporaryDirectory() as temp_dir :

    private_tab, common_tab = st.tabs(["Private photos", "Common photos"])

    with private_tab:
        
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name
        chosen_dir = temp_dir
        if "camera_dis" not in st.session_state.keys():
            st.session_state['camera_dis'] = False
    
    
        upload_option = st.selectbox(
       "How would you like to upload photos?",
       ("Files", "Camera"),
       index=1, key = 1)
        
        if 'time' not in st.session_state:
            st.session_state['time'] =  str(int(time.time()))
            #os.mkdir(f"panorama/{st.session_state['time']}")
            #temp_folder = f"panorama/{st.session_state['time']}"
        def find_imgs(folder):
            files = os.listdir(folder)
            images = [x for x in files if ".jpg" in x]
            images.sort()
            return images
            
            
        if upload_option == "Camera":

            
                
        
            picture = st.camera_input("Take a picture", disabled=st.session_state['camera_dis'], key = 2)
        
            if "image_counter" not in st.session_state:
                st.session_state['image_counter'] = 0
            if "taken_images" not in st.session_state:
                st.session_state['taken_images'] = []
    
            
            if picture:
                st.session_state['taken_images'].append(picture)
            for i,picture in enumerate(st.session_state['taken_images']):
                with open(os.path.join(temp_dir,f"{i}.jpg"),"wb") as f: 
                    f.write(picture.getbuffer())         
                st.session_state['image_counter'] += 1
                #st.write(os.listdir(temp_dir))
                
    
        elif upload_option == "Files" :
            uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, key = 40)
            for picture in uploaded_files:
                with open(os.path.join(temp_dir,picture.name),"wb") as f: 
                    f.write(picture.getbuffer())         
                
    
        
    

        st.subheader("Private photos")
        st.write(f":red[{len(find_imgs(temp_dir))}] :frame_with_picture:")

        if len(find_imgs(temp_dir)) > 0:

            if st.button('Show images', key = 3):
                columns = st.columns(len(find_imgs(temp_dir)))
                for i, col in enumerate(columns):
                    with col:
                        st.image(f'panorama/tempDir/{find_imgs(temp_dir)[i]}')
            if st.button('Delete all', key = 4):
                directory_path = temp_dir
                try:
                    delete_files(directory_path)
                except:
                    pass

    
    
        if len(find_imgs(chosen_dir)) >= 2:
            c1, col_main, c3 = st.columns(3)
            with c1:
                pass
            with col_main:
                center_button = st.button('Lets warp!', key = 5)
            with c3:
                pass
            if center_button:
                st.session_state['camera_dis'] = True
                # ------------------------------------------------------------------------------------------
                # get pano images
                # ------------------------------------------------------------------------------------------
          
                # Create an ImageCollection from the uploaded images
                pano_image_collection = io.ImageCollection(f"{chosen_dir}/*",load_func=lambda f: io.imread(f).astype(np.float64) / 255)
        
    
        
                # ------------------------------------------------------------------------------------------
                # example with 2 images descriptors
                # ------------------------------------------------------------------------------------------
        
                src, dest = pano_image_collection[0], pano_image_collection[1]
                src_keypoints, src_descriptors = panoramafunc.find_keypoints(src)
                dest_keypoints, dest_descriptors = panoramafunc.find_keypoints(dest)
        
                robust_transform, matches = panoramafunc.ransac_tr(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, return_matches=True)
        
                plots.plot_inliers(src, dest, src_keypoints, dest_keypoints, matches)
        
                # ------------------------------------------------------------------------------------------
                # point and descriptors for all images pairs
                # ------------------------------------------------------------------------------------------
        
                keypoints, descriptors = zip(*(panoramafunc.find_keypoints(img) for img in pano_image_collection))
    
                forward_transforms = tuple(panoramafunc.ransac_tr(sorc_kp, src_desc, dest_kp, dest_desc)
                                            for sorc_kp, src_desc, dest_kp, dest_desc
                                            in zip(keypoints[:-1], descriptors[:-1], keypoints[1:], descriptors[1:]))
        
                # ------------------------------------------------------------------------------------------
                # Merge and find borders
                # ------------------------------------------------------------------------------------------    
        
        
        
                simple_center_warps = panoramafunc.find_center_warps(forward_transforms)
                corners = np.flip(tuple(panoramafunc.get_corners(pano_image_collection, simple_center_warps)), axis= None)
                min_coords, max_coords = panoramafunc.get_min_max_coords(corners)
                central_img = pano_image_collection[ (len(pano_image_collection) - 1) // 2]
        
                plots.plot_warps(corners, min_coords=min_coords, max_coords=max_coords, img=central_img)
    
                # ------------------------------------------------------------------------------------------
                # Get final wrap
                # ------------------------------------------------------------------------------------------
        
                final_central_warps, output_shape = panoramafunc.get_final_center_warps(pano_image_collection, simple_center_warps)
                corners = np.flip(tuple(panoramafunc.get_corners(pano_image_collection, final_central_warps)), axis= None)
        
        
                result_pano = panoramafunc.merge_final_pano(pano_image_collection, final_central_warps, output_shape)
        
                plots.plot_result(result_pano)
        
                # ------------------------------------------------------------------------------------------
                # Get smoothed pano
                # ------------------------------------------------------------------------------------------
                
                result_pano_smth = panoramafunc.gaussian_merging_pano(pano_image_collection, final_central_warps, output_shape)
                      
                plots.plot_result(result_pano_smth, title = "Smoothed pano")
                
                temp_dir_obj.cleanup()
                if "taken_images" in st.session_state:
                    del st.session_state['taken_images']
                if st.button('Reset', key = 6):
                    directory_path = temp_dir
                    try:
                        delete_files(directory_path)
                    except:
                        pass

    with common_tab:
        
 
        chosen_dir = "panorama/tempDir"
        if "camera_dis" not in st.session_state.keys():
            st.session_state['camera_dis'] = False
    
    
        upload_option = st.selectbox(
       "How would you like to upload photos?",
       ("Files", "Camera"),
       index=1, key = 7)
        
        if 'time' not in st.session_state:
            st.session_state['time'] =  str(int(time.time()))
            #os.mkdir(f"panorama/{st.session_state['time']}")
            #temp_folder = f"panorama/{st.session_state['time']}"
        def find_imgs(folder):
            files = os.listdir(folder)
            images = [x for x in files if ".jpg" in x]
            images.sort()
            return images
            
            
        if upload_option == "Camera":


                
            picture_com = st.camera_input("Take a picture", disabled=False, key = 8)
            if "image_counter" not in st.session_state:
                st.session_state['image_counter'] = 0

    
            if picture_com:
                with open(os.path.join("panorama/tempDir",f"{str(int(time.time()))}.jpg"),"wb") as f: 
                    f.write(picture_com.getbuffer())         
                st.session_state['image_counter'] += 1

            
                
    
        elif upload_option == "Files" :
            uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, key = 41)
            for picture in uploaded_files:
                with open(os.path.join("panorama/tempDir",picture.name),"wb") as f: 
                    f.write(picture.getbuffer())

        st.subheader("Common photos")
        st.write(f":blue[{len(find_imgs('panorama/tempDir'))}] :frame_with_picture:")
                    
        if len(find_imgs('panorama/tempDir')) > 0:

            if st.button('Show images', key = 9):
                columns = st.columns(len(find_imgs("panorama/tempDir")))
                for i, col in enumerate(columns):
                    with col:
                        st.image(f'panorama/tempDir/{find_imgs("panorama/tempDir")[i]}')
            if st.button('Delete all', key = 10):
                directory_path = "panorama/tempDir"
                try:
                    delete_files(directory_path)
                except:
                    pass
    
        
    


    
    
        if len(find_imgs(chosen_dir)) >= 2:
            c1, col_main, c3 = st.columns(3)
            with c1:
                pass
            with col_main:
                center_button = st.button('Lets warp!', key = 14)
            with c3:
                pass
            if center_button:
                st.session_state['camera_dis'] = True
                # ------------------------------------------------------------------------------------------
                # get pano images
                # ------------------------------------------------------------------------------------------
          
                # Create an ImageCollection from the uploaded images
                pano_image_collection = io.ImageCollection(f"{chosen_dir}/*",load_func=lambda f: io.imread(f).astype(np.float64) / 255)
        
    
        
                # ------------------------------------------------------------------------------------------
                # example with 2 images descriptors
                # ------------------------------------------------------------------------------------------
        
                src, dest = pano_image_collection[0], pano_image_collection[1]
                src_keypoints, src_descriptors = panoramafunc.find_keypoints(src)
                dest_keypoints, dest_descriptors = panoramafunc.find_keypoints(dest)
        
                robust_transform, matches = panoramafunc.ransac_tr(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, return_matches=True)
        
                plots.plot_inliers(src, dest, src_keypoints, dest_keypoints, matches)
        
                # ------------------------------------------------------------------------------------------
                # point and descriptors for all images pairs
                # ------------------------------------------------------------------------------------------
        
                keypoints, descriptors = zip(*(panoramafunc.find_keypoints(img) for img in pano_image_collection))
    
                forward_transforms = tuple(panoramafunc.ransac_tr(sorc_kp, src_desc, dest_kp, dest_desc)
                                            for sorc_kp, src_desc, dest_kp, dest_desc
                                            in zip(keypoints[:-1], descriptors[:-1], keypoints[1:], descriptors[1:]))
        
                # ------------------------------------------------------------------------------------------
                # Merge and find borders
                # ------------------------------------------------------------------------------------------    
        
        
        
                simple_center_warps = panoramafunc.find_center_warps(forward_transforms)
                corners = np.flip(tuple(panoramafunc.get_corners(pano_image_collection, simple_center_warps)), axis= None)
                min_coords, max_coords = panoramafunc.get_min_max_coords(corners)
                central_img = pano_image_collection[ (len(pano_image_collection) - 1) // 2]
        
                plots.plot_warps(corners, min_coords=min_coords, max_coords=max_coords, img=central_img)
    
                # ------------------------------------------------------------------------------------------
                # Get final wrap
                # ------------------------------------------------------------------------------------------
        
                final_central_warps, output_shape = panoramafunc.get_final_center_warps(pano_image_collection, simple_center_warps)
                corners = np.flip(tuple(panoramafunc.get_corners(pano_image_collection, final_central_warps)), axis= None)
        
        
                result_pano = panoramafunc.merge_final_pano(pano_image_collection, final_central_warps, output_shape)
        
                plots.plot_result(result_pano)
        
                # ------------------------------------------------------------------------------------------
                # Get smoothed pano
                # ------------------------------------------------------------------------------------------
                
                result_pano_smth = panoramafunc.gaussian_merging_pano(pano_image_collection, final_central_warps, output_shape)
                      
                plots.plot_result(result_pano_smth, title = "Smoothed pano")
                
                temp_dir_obj.cleanup()
                if "taken_images" in st.session_state:
                    del st.session_state['taken_images']
                if st.button('Reset', key = 11):
                    directory_path = temp_dir
                    try:
                        delete_files(directory_path)
                    except:
                        pass


elif selected == 'Image generation':
    st.markdown('<p style="font-family: Pragmatica; font-size: 55px;">Cars image generation</p>', unsafe_allow_html=True)
    n_cars = st.slider('Cars to generate',min_value=1, max_value=10)
    if st.button('Lets generate!'):
        show_gan_imgs(n_cars)
        

