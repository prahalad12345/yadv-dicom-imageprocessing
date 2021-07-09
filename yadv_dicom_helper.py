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
# This file includes all helper functions for DICOM Operations
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

def load_file(path):
    images=[dicom.read_file(files) for files in (path)]
    
    images.sort(key=lambda x:int(x.InstanceNumber))
    slicethickness=np.abs(images[0].ImagePositionPatient[2]-images[1].ImagePositionPatient[2])
    for files in images:
        files.SliceThickness=slicethickness
    return images


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



def dicomProcessing():
    st.subheader("DICOM Operation")
    uploaded_file = st.file_uploader("Upload a DICOM image file", type=[".dcm"])
    col1, col2 = st.beta_columns(2)

    if uploaded_file:
        file_bytes = dicom.read_file(uploaded_file)
        image = file_bytes.pixel_array
        image=cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        col1.subheader("Original Image")
        col1.image(image)
        df = create_table(file_bytes)
        st.write(df)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if(file_bytes.Modality=='CT'):
            imagetoplot=gethu(file_bytes)
            leftcol,rightcol = st.beta_columns(2)
            leftcol.pyplot(hounsfieldunitgraphh(imagetoplot))
            rightcol.write(hudataframe())
        try:
            elem=file_bytes[0x6000, 0x3000]
        except:
            array=np.zeros((len(image),len(image[0])))
            packed_bytes=pack_bits(array)
            file_bytes.add_new(0x60003000,'OW',packed_bytes)
                        

    # types
    type_name = st.sidebar.selectbox("Choose DICOM operation", ["Windowing",
                                                                "Zoom",
																"Rotate",
																"Overlays"])
    dict={"Windowing":windowing,"Zoom":handle_zoom,"Rotate":rotation,"Overlays":overlay}
    try:
        if st.button('Export'):
            exportbutton(file_bytes)
        if type_name == "Zoom" or type_name=="Rotate": 
            col2.subheader(type_name)	
            res=dict[type_name](image)
            col2.image(res)
        elif type_name == "Windowing" and file_bytes.Modality=='CT':
            col2.subheader(type_name)
            res=dict["Windowing"](file_bytes)
            col2.image(res)
        elif type_name == "Overlays":
            col2.subheader(type_name)
            file=dict['Overlays'](file_bytes,image)
            col2.image(file)
            if(st.sidebar.button('Export Overlay image')):
                exportoverlay(file_bytes,file)
        else:
            st.error("Choose an appropriate option")
    except UnboundLocalError as error:
        # Output expected UnboundLocalErrors.
        st.error(error)
    except Exception as exception:
        # Output unexpected Exceptions.
        st.error(exception, False)	
    
def handle_zoom(image):		
    try:
        zoomslider = st.sidebar.slider('Zoom In/Out', -9, 15,0)
        if(zoomslider<0):
            adjustment=(float)(10+zoomslider)/10
            half = cv2.resize(image, (0, 0), fx =adjustment , fy = adjustment)
            return half
        elif(zoomslider==0):
            return image 
        else:
            adjustment=(float)(10+zoomslider)/10
            big = cv2.resize(image, (0, 0), fx =adjustment , fy = adjustment)
            return big
    except UnboundLocalError as error:
        # Output expected UnboundLocalErrors.
        st.error(error)
    except Exception as exception:
        # Output unexpected Exceptions.
        st.error(exception, False)
	
def rotation(image):
    rotateslider = st.sidebar.slider('Angle of rotation', 0,359)
    centerX=(image.shape[0])/2
    centerY=(image.shape[1])/2
    angleofrotation=rotateslider*np.pi/180.0
    cosTheta = np.cos(angleofrotation)
    sinTheta = np.sin(angleofrotation)
    tx = (1-cosTheta) * centerX - sinTheta * centerY
    ty =  sinTheta * centerX  + (1-cosTheta) * centerY
    warpMat = np.float32(
    [
        [ cosTheta, sinTheta, tx],
        [ -sinTheta,  cosTheta, ty]
    ])
    outDim = image.shape[0:2]
    result = cv2.warpAffine(image, warpMat, outDim)
    return result


def windowing(img):
    hu1slider = st.sidebar.slider('Minimum HU value', -1000,2999)
    hu2slider = st.sidebar.slider('Maximum HU value', -1000,3000)

    intercept, slope = get_windowing(img)
    image =img.pixel_array
    finalimage = window_image(image, hu1slider, hu2slider, intercept, slope)
    return finalimage
		
def create_table(ds):
    flag=0
    for i in ds.keys():
        flag=i
    keywords=[]
    finale=[]
    patientdata = open('patientdata.txt' , 'w' )
    for i in ds.keys():
        if(ds[i].VR == "SQ"):
            continue
        if(i!=flag):
        #print(ds[i].keyword)
            keywords.append(str(i))
            keywords.append(ds[i].keyword)
            
            keywords.append(ds[i].value)
            finale.append(keywords)
            newline='\n'
            patientdata.write(str(i)+" "+(str)(ds[i].keyword)+" "+(str)(ds[i].value)+newline)
            keywords=[]
    df = pd.DataFrame(finale , columns = ['(Group, Tag)','Description','Value'])

    patientdata.close()
    return df

def gethu(dicomfile):
    im=dicomfile.pixel_array
    im=im.astype(np.int16)
    im[im==-2000] = 0
    intercept=dicomfile.RescaleIntercept
    slope=dicomfile.RescaleSlope

    im = slope * im.astype(np.float64)
    im = im.astype(np.int16)

    im+=np.int16(intercept)
    
    return im

def hounsfieldunitgraph(image):
    im=image.astype(np.float64)
    plt.hist(im.flatten(),bins=40,color='c')
    plt.xlabel('Hounsfield unit')
    plt.ylabel=('frequency')
    fig=plt.figure()
    return fig    
 
def hounsfieldunitgraphh(image):
    im=image.astype(np.float64)
    plt.hist(im.flatten(),bins=40,color='c')
    plt.xlabel('Hounsfield unit')
    plt.ylabel=('frequency') 

def hudataframe():
    hulist=[['Air','-1000'],['Lung','-500'],['Fat','-100 to -50'],['Water','0'],['CSF','15'],['Kidney','30'],['Blood','30 to 45'],['Muscle','10 to 40'],['whitematter','20 to 30'],['greymatter','37 to 45'],['Liver','40 to 60'],['Softtissue','100 to 300'],['Bone','700 (cancellous bone) to 3000 (cortical bone']]
    df=pd.DataFrame(hulist,columns=['Substance','HU value'])
    return df
    
def load_scan(path):
    files=[dicom.read_file(dicomfile) for dicomfile in path]
    
    
    files.sort(key=lambda x:int(x.InstanceNumber))
    images=[]
    for im in files:
        image = im.pixel_array
        image=cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        images.append(image)
    
    return images

def window_image(img, window_center,window_width, intercept, slope, rescale=True):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    
    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)
    
    return img

def get_first_of_dicom_field_as_int(x):
    #To prevent overlap if multiple images are present in the same dicom file
    if type(x) == dicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    dicom_fields = [ data.RescaleIntercept,
                    data.RescaleSlope]
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def exportbutton(file):
    file.PatientName=sha256(file.PatientName)
    file.PatientSex=sha256(file.PatientSex)
    file.save_as('Exportfile/'+(str)(file.PatientName)+'-'+(str)(file.InstanceNumber) + ".dcm")
    st.success("file saved")

    return

def overlay(ds,image):
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ",'#ffffff')
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")

    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
    )
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    cv2.imwrite('bemp.png',image)
    imagee=Image.open("bemp.png")
    
	# Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=imagee,
        update_streamlit=realtime_update,
        height=len(image),
        width=len(image[0]),
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # Do something interesting with the image data and paths
    if canvas_result.json_data is not None:
        st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))

    im=canvas_result.image_data[: , : , 0:3]     
    
    grayscale=0.299*im[:,:,2] + 0.587*im[:,:,1] + 0.114*im[:,:,0]
    storage=grayscale

    for i in range(len(grayscale)):
        for j in range(len(grayscale[0])):
            if(grayscale[i][j]>=125):
                storage[i][j]=1
            else:
                storage[i][j]=0

    storage=storage.astype(np.uint8)

    ds[0x6000,0x3000].value=pack_bits(storage)
    storage*=255

    imagefinal=cv2.add(image,storage)
    return imagefinal

def exportoverlay(file,image):
    name=sha256(file.PatientName)
    cv2.imwrite('overlayimage/'+(str)(name)+'-'+(str)(file.InstanceNumber)+".jpg",image)
    st.success("file saved")