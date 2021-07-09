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
# Core Capbilities
# -----------------
# Minimum Viable Product (MVP)
#   -1-  Load and View DICOM Image
#   -2-  View DICOM Head Metadata Information
#   -3-  Basic DICOM Image Operation – i.e. Rotation, Zoom, Measurement etc.
#   -4-  DICOM Image Classification
#   -5-  Export DICOM Image – Anonymize Personal Information
#   -6-  COVID analysis using CT images and XRay Images
#   -7-  Advance measurement overlay on top of images
#   -8- More 3D Image support
#   -9- Open multiple image in parallel and compare - Advanced Diagnosis
#
# Upcoming Features
#   -7-  Hand Gesture based Image operation
#   -8-  The Value Of Interest Lookup (VOI LUT - 0028, 3010) Table Implementation
#   -9-  Advance measurement overlay on top of images
#   -10- Advanced 3D Operation support
#   -11- Open multiple image in parallel and compare - Advanced Diagnosis
#   -12- Integrate with Integration with PACS
#	-13- Scan patient, auto face detect and pull patient history from PACS
#
from yadv_dicom_helper import *
from yadv_3ddicom_helper import *
from yadv_covid_analyzer import *
from yadv_multidicom_analyzer import *
from PIL import Image

def main():
    # YaDV Application setup
    st.header("YaDV - Yet Another DICOM Viewer")
    icon = Image.open('./images/YADV.png')
    st.sidebar.image(icon, caption="Yet another DICOM Viewer")
    st.sidebar.title("YaDV - Options")
	
    # YaDV - Main menu
    # Core capabilities to YaDV will be added to this
    function_selected = st.sidebar.selectbox("Choose YaDV operations",
                                             ["Multiple DICOM",
											  "DICOM Operations",
                                              "3D Plotting",
                                              "Covid Detector"])
									  
    # YaDV - Main menu action handler
    dict={"Multiple DICOM":multipleDicom,"DICOM Operations":dicomProcessing,"3D Plotting":threedVisulaization,"Covid Detector":covidDetector}	
    dict[function_selected]()

    # YaDV product summary
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app is maintained by Prahalad. YaDV is simple DICOM viewer 
		for medical images designed to provide users with simple experience
		to view and analyze DICOM images..
        """)

if __name__ == "__main__":
    main()
