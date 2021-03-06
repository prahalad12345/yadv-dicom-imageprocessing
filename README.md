# Welcome to YaDV (Yet another DICOM Viewer)

**Platform to open, view, analyze, classify and enrich DICOM format images..**

## Glossary
| Term      | Description |
| ----------- | ----------- |
| Radiology      | Radiology is a branch of medicine that uses imaging technology to diagnose and treat disease |
| Radiologists   | Radiologist are medical doctors that specialize in diagnosing and treating injuries and diseases using medical imaging (radiology) procedures (exams/tests) such as X-rays, computed tomography (CT), ultrasound etc.|
| PACS | **P**icture **A**rchiving and **C**ommunication **S**ystem: PACS is a high-speed, graphical, computer network system for the storage, recovery, and display of radiologic images|
| HL7 | **H**ealth **L**evel Seven International : HL7 is a set of international standards used to transfer and share data between various healthcare providers |
| DICOM| **D**igital **I**maging and **CO**mmunications in **M**edicine – an universal standard for Digital Imaging |
| NEMA| **N**ational **E**lectronic **M**anufacturing **A**ssociation - An ANSI-accredited Standards Developing Organization. The DICOM Standard is managed by the Medical Imaging & Technology Alliance - a division of the NEMA.|

## About YaDV
YaDV is a web-based DICOM Viewer for PACS enables Radiologist to diagnoses, viewing, and transmitting medical images. 

## What problem will YaDV solves?
* YaDV is an DICOM viewer greatly facilitates the day-to-day of cardiologists, traumatologist, oncologists, etc. and most importantly, it improves the healthcare and service to patients.
* YaDV support different image processing abilities and advanced functions. Besides image visualisation YaDV can take measurements and convert images to other formats.
* YaDV provides 3D image viewing for the surgical planning
What makes YaDV different: One stop solution for COVID Analysis DICOM format, 3D object view, Image Classification and in future will add hand gesture based DICOM image processing.

## DICOM Overview
* DICOM stands for Digital Imaging and COmmunications in Medicine – an universal standard for Digital Imaging.
* DICOM is a specification for the creation, transmission, and storage of digital medical image and report data. 
* Another important acronym that seemingly all DICOM vendors plug into their names is PACS (Picture Archiving and Communication Systems).
* PACS are medical systems (consisting of necessary hardware and software) built to run digital medical imaging. They comprise:
* - Modalities: Digital image acquisition devices, such as CT scanners or ultrasound.
* - Digital image archives: Where the acquired images are stored.
* - Workstations: Where radiologists view (“read”) the images. (YaDV)

## YaDV Architecture
![YaDV Architecture](https://github.com/prahalad12345/yadv-dicom-imageprocessing/blob/d59ea81a3effaf9aedf32e620340776292bbfefa/images/yadvArchitecture.png)

## YaDV Module Structure
![YaDV Architecture](https://github.com/prahalad12345/yadv-dicom-imageprocessing/blob/d59ea81a3effaf9aedf32e620340776292bbfefa/images/yadvModule.png)

## Core Capabilities
1. Load and View DICOM Image
2. View DICOM Head Metadata Information
3. Basic DICOM Image Operation – i.e. Rotation, Zoom, Measurement etc.
4. Image Classification
5. 3D image rendering using Marching Cube
6. Anonymize during export
7. Cloud Deployed (Demo purpose deployed in AWS EC2 instance)

## Prerequisites
Make sure you have installed all of the following prerequisites on your development machine:

* Git - [Download & Install Git](https://git-scm.com/downloads). OSX and Linux machines typically have this already installed.
* Python - [Download & Install Python3](https://www.geeksforgeeks.org/download-and-install-python-3-latest-version/) - Minimum requirement 3.8.x


## Installation
YaDV is based on streamlit and streamlit can also be installed in a virtual environment on Windows, Mac and Linux. 

```bash
pip install -r requirement.txt
```

[OR]


```bash
pip install streamlit
pip install numpy
pip install fastai
pip install -Uqq fastbook
pip install pytorch
pip install streamlit-drawable-canvas
pip install cryptohash
pip install pandas
pip install pydicom
pip install opencv-python
pip install skimage
pip install matplotlib
pip install plotly
pip install mplot3d-dragger
pip install graphviz
pip install scikit-image
```

## Running YaDV
YaDV can be executed used following command:

```bash
cd yadv-dicom-imageprocessing
mkdir overlayimage
mkdir Exportfile
streamlit run yadv.py
```

## Development

```bash
git clone https://github.com/prahalad12345/yadv-dicom-imageprocessing.git
```

## 3rd Party Library Dependencies 
YaDV uses following 3rd party tools/libraries:

| 3rd Party      | Reference Link |
| ----------- | ----------- |
| OpenCV | https://opencv.org/|
| PyDICOM | https://pydicom.github.io/ |
| Fast.ai | https://www.fast.ai/ |
| Streamlit | https://streamlit.io/ |
| Numpy | https://numpy.org/ |
| Pandas | https://pandas.pydata.org/ |
| Matplot | https://matplotlib.org/ |
| Mplot3d | https://matplotlib.org/ |
| Plotly | https://plotly.com/ |
| SkImage | https://scikit-image.org/ |
| Cryptohash | https://www.cryptohash.net/ |

## Next Step
1. The Value Of Interest Lookup (VOI LUT - 0028, 3010) Table Implementation
2. Hand Gesture based Image operation - End Goal (New viewer different from YaDV tech stack)
3. Open multiple image in parallel and compare
4. Predict COVID cases based on CT Images
5. Predict COVID cases based on XRays
6. Integration with PACS
7. Scan patient, auto face detect and pull patient history from PACS
8. Advance measurement overlay on top of images

## Reference

| Topic      | Reference Link |
| ----------- | ----------- |
| Digital Image Communication in Medicine | Book from Oleg S. Pianykh |
| National Electrical Manufacturers Association (NEMA) | http://dicom.nema.org/medical/dicom/current/output/chtml/part10/chapter_7.html |
| Python library for DICOM | https://pypi.org/project/pydicom/  |
| FAST.AI for Medical Image Processing | https://docs.fast.ai/medical.imaging |
| User Interface for Image Viewer | https://streamlit.io/ |
| Heroku Deployment | https://towardsdatascience.com/deploying-a-basic-streamlit-app-to-heroku-be25a527fcb3 |
| AWS Deployment | https://towardsdatascience.com/how-to-deploy-a-streamlit-app-using-an-amazon-free-ec2-instance-416a41f69dc3|

## FAQ
1.  If you get "ImportError: libGL.so.1: cannot open shared object file" error message install following package:
    ```bash
    sudo apt install libgl1-mesa-glx
    ```

## License

YaDV is completely free and open-source 
