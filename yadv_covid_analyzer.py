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
#
# YaDV COVID Analyzer code
#
import fastbook
from fastbook import *
import matplotlib.pyplot as plt
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
from streamlit import uploaded_file_manager

def covidDetector():
    st.subheader("COVID Detector")
    uploaded_file = st.file_uploader("Upload a front lung scan image ")
    #filee=uploaded_file
    if uploaded_file:
        print(type(uploaded_file.type))
        if uploaded_file.type == 'application/dicom' or uploaded_file.type=='application/octet-stream':
            file_bytes = dicom.read_file(uploaded_file)
            img = file_bytes.pixel_array
            img=cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img=cv2.resize(img,(2500,2500),interpolation = cv2.INTER_AREA)
        st.image(img)
    learn_inf = load_learner('export.pkl')
    try:
        s=learn_inf.predict(cv2.resize(img,(128,128)))
        st.success(s[0])
    except UnboundLocalError as error:
        # Output expected UnboundLocalErrors.
        st.error(error)
    except Exception as exception:
        # Output unexpected Exceptions.
        st.error(exception, False)