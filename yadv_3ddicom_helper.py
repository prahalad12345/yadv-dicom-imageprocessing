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
# YaDV DICOM Helper Utils for 3d image processing
#
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
  

def threedVisulaization():
    st.subheader("DICOM Operation")
    uploaded_file = st.file_uploader("Upload a DICOM image file",accept_multiple_files=True)
    if uploaded_file:
        st.subheader("DICOM Image List:")
        imagelist = load_file(uploaded_file)
        count=0
        
    type_name = st.sidebar.selectbox("Choose DICOM operation", ["Plot 3d","Interactive 3d"])

    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    try:
        for i in range(len(imagelist)):
            if(imagelist[i].Modality!='CT'):
                st.error('This section is for CT images')
                return
        imagee=getPixelHounsFieldUnit(imagelist)
        pixresampled,spacing = resampleDICOMImage(imagee,imagelist,[1,1,1])

        v,f=create3dMesh(pixresampled,350,2)
        dict={"Plot 3d":plt_3d,"Interactive 3d":plotly_3d}
        if type_name == "Plot 3d" :
            st.pyplot(dict[type_name](v,f))
        elif type_name == "Interactive 3d" :
            st.plotly_chart(dict[type_name](v,f))
        else:
            st.info("Choose right option")	
    except UnboundLocalError as error:
        # Output expected UnboundLocalErrors.
        st.error(error)
    except Exception as exception:
        # Output unexpected Exceptions.
        st.error(exception, False)
    except:
        # Output expected UnboundLocalErrors.
         st.error("error")

def getPixelHounsFieldUnit(scans):
    image=np.stack([files.pixel_array for files in scans])

    image=image.astype(np.int16)
    
    image[image==-2000] = 0
    
    intercept=scans[0].RescaleIntercept
    slope=scans[0].RescaleSlope
    
    if slope != 1:
        image=slope*image.astype(np.float64)
        image=image.astype(np.int16)
    image+=np.int16(intercept)
    
    return np.array(image , dtype=np.int16)

def resampleDICOMImage(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + (list)(scan[0].PixelSpacing), dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def create3dMesh(image, threshold=-300, step_size=1):
    p = image.transpose(2,1,0)
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces

def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    
    # Make the colormap single color since the axes are positional not intensity. 
    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
    
    fig = FF.create_trisurf(x=x,
                        y=y, 
                        z=z, 
                        plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization")
    return fig

def plt_3d(verts, faces):
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_facecolor((0.7, 0.7, 0.7))
	
def load_file(path):
    images=[dicom.read_file(files) for files in (path)]
    if(images[0].Modality!='CT'):
        return images
    images.sort(key=lambda x:int(x.InstanceNumber))

    slicethickness=np.abs(images[0].ImagePositionPatient[2]-images[1].ImagePositionPatient[2])
    
    for files in images:
        files.SliceThickness=slicethickness
    return images	

