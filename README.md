# LLDNet: A Lightweight Lane Detection Approach for Autonomous Cars Using Deep Learning

This is the source code of LLDNet: A Lightweight Lane Detection Approach for Autonomous Cars Using Deep Learning. We provide the dataset, model development code, test code and the pretrained model.

Khan, M.A.-M.; Haque, M.F.; Hasan, K.R.; Alamjani, S.H.; Baz, M.; Masud, M.; Al-Nahid, A. LLDNet: A Lightweight Lane Detection Approach for Autonomous Cars Using Deep Learning. Sensors 2022, 22, 5595. https://doi.org/10.3390/s22155595

# Network Architecture
![image](https://user-images.githubusercontent.com/33350185/181172871-77060790-4437-44c9-ba26-d88ee953e114.png)

# Some Results
![image](https://user-images.githubusercontent.com/33350185/181173323-d740d99e-29f1-4bad-946c-57e10f17db11.png)
![image](https://user-images.githubusercontent.com/33350185/181174138-31ac678a-080f-44eb-9ad0-2b0dac4efcb7.png)
![image](https://user-images.githubusercontent.com/33350185/181174183-7788e669-e02e-4215-b771-9319c78a31fe.png)

# Dataset
## Description 
In this work, we constructed a mixed dataset by combining the “Udacity Machine Learning Nanodegree Project Dataset” and the “Cracks and Potholes in Road Images Dataset”. The first dataset was collected by a smartphone and contained 12,764 training images. The images were extracted from 12 different videos filmed at 30 fps. The speciality of this dataset is that it contains images of different weather conditions, different road curvatures, and different lighting conditions. The other dataset of our work collected images by using a Highway Diagnostic Vehicle (DHV). This dataset contains 2235 images which were extracted from a few videos filmed at 30 fps with. The speciality of this dataset is that most of the images in this dataset contain images with cracks and holes on the road. However, the roads on the images are not heavily damaged; rather, the images contain roads with minor damage. Each image of this dataset contains three mask images, including lane marking, hole marking, and crack marking masks. However, we only consider the lane marking masks images of this dataset. The primary purpose of mixing these two datasets in our work is that we want to develop a robust system. The system can detect the lanes in adverse weather or lighting conditions, even if the roads are defective and unstructured.

## Download
The mixed dataset utilized in this work can be found https://drive.google.com/file/d/1S23Ac0_hbOktV0rE2q0IkQWpQjUfkMTB/view?usp=sharing 
and https://drive.google.com/file/d/1I264WVBL3Dyp_4PTfEYkVIDkg_Yn5gJJ/view?usp=sharing
