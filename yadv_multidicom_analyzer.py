#       YYYYYYY       YYYYYYY                DDDDDDDDDDDDD      VVVVVVVV           VVVVVVVV
#       Y:::::Y       Y:::::Y                D::::::::::::DDD   V::::::V           V::::::V
#       Y:::::Y       Y:::::Y                D:::::::::::::::DD V::::::V           V::::::V
#       Y::::::Y     Y::::::Y                DDD:::::DDDDD:::::DV::::::V           V::::::V
#       YYY:::::Y   Y:::::YYYaaaaaaaaaaaaa     D:::::D    D:::::DV:::::V           V:::::V 
#          Y:::::Y Y:::::Y   a::::::::::::a    D:::::D     D:::::DV:::::V         V:::::V  
#           Y:::::Y:::::Y    aaaaaaaaa:::::a   D:::::D     D:::::D V:::::V       V:::::V   
#            Y:::::::::Y              a::::a   D:::::D     D:::::D  V:::::V     V:::::V    
#             Y:::::::Y        aaaaaaa:::::a   D:::::D     D:::::D   V:::::V   V:::::V     
#              Y:::::Y       aa::::::::::::a   D:::::D     D:::::D    V:::::V V:::::V      
#              Y:::::Y      a::::aaaa::::::a   D:::::D     D:::::D     V:::::V:::::V       
#              Y:::::Y     a::::a    a:::::a   D:::::D    D:::::D       V:::::::::V        
#              Y:::::Y     a::::a    a:::::a DDD:::::DDDDD:::::D         V:::::::V         
#           YYYY:::::YYYY  a:::::aaaa::::::a D:::::::::::::::DD           V:::::V          
#           Y:::::::::::Y   a::::::::::aa:::aD::::::::::::DDD              V:::V           
#           YYYYYYYYYYYYY    aaaaaaaaaa  aaaaDDDDDDDDDDDDD                  VVV            
#
# YaDV is a web-based DICOM Viewer for PACS enables user to 
# diagnoses, viewing, and transmitting medical images. 
# YaDV DICOM Helper Utils
#
# This file includes all helper functions for multiple DICOM image processing
#
import matplotlib.pyplot as plt
from numpy.lib.type_check import typename
from scipy.ndimage.interpolation import zoom
import streamlit as st
import numpy as np
import pandas as pd
import pydicom as dicom
import os
import cv2
import numpy as np
import pandas as pd 
import pydicom as dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure,morphology
from skimage import exposure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
from plotly.tools import FigureFactory as FF
from plotly.offline import plot,iplot
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from cryptohash import sha256
from streamlit_drawable_canvas import st_canvas
from PIL import Image,ImageOps
from pydicom.pixel_data_handlers.numpy_handler import pack_bits

def multipleDicom():

    '''

        multiple dicom function helps the physician to view multiple image at once 
        This option is required when there is a requirement of working with multiple images at once
        There are two ways of viewing : 1)viewing each image using a slider
                                        2)viewing all images at once

    '''

    st.subheader("DICOM Operation")
    uploaded_file = st.file_uploader("Upload a DICOM image file",accept_multiple_files=True , type=[".dcm"])

    if uploaded_file:
        st.subheader("DICOM Image List:")
        
        image = load_scan(uploaded_file)
    type_name = st.sidebar.selectbox("Choose DICOM operation", ["Slide Images",
                                                                "Multiple Image At Once"])
    try:
        if(len(image)==1 or len(image)==0):
            '''
                if there is only 1 image or no images then this error message is printed .
                if this codeblock doesnt exist and ugly exception message is printed
            '''
            st.error("expected more than 1 image")
        else:
                if type_name == "Slide Images": 
                    '''
                        execution of the 1) option from the above docstring
                    '''
                    st.subheader("Slide Images")	
                    res=slideimage(image)
                    st.image(res)
                elif type_name == "Multiple Image At Once":
                    '''
                        execution of the 2) option from the above docstring
                    '''
                    col1,col2,col3=st.beta_columns(3)
                    for i in range(len(image)):
                        if(i%3==0):
                            col1.image(image[i])
                        if(i%3==1):
                            col2.image(image[i])
                        if(i%3==2):
                            col3.image(image[i])
                else:
                    '''
                        this is an optional block since there is bounding of  two options to choose
                    '''
                    st.info("Choose right option")	
    except UnboundLocalError as error:
        '''
            Output expected UnboundLocalErrors.
            handling the error when there is no dicom file present in the upload section
        '''
        st.error(error)
    except Exception as exception:
        # Output unexpected Exceptions.
        st.error(exception, False)	
		
def slideimage(images):
    '''
        displaying the image at that respective index which the slider is pointing to 
    '''
    if(len(images)==1):
        st.error('require more than 1 image')
        return 0
    else:
        imageslider = st.sidebar.slider('image No', 1, len(images))

        return images[imageslider-1]

def load_scan(path):

    '''
        This function collects the image from the dicom file
        The image is also sorted based on its InstanceNumber
    '''

    files=[dicom.read_file(dicomfile) for dicomfile in path]
    files.sort(key=lambda x:int(x.InstanceNumber))
    images=[]
    for im in files:
        image = im.pixel_array
        image=cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        images.append(image)
    
    return images
		