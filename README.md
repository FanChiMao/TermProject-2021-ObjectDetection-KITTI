# TermProject-2021-kitti  

## 1. Dataset Preparation    
- [KITTI dataset](https://drive.google.com/drive/folders/1f47HB5gLQDElIAf600SX-VtrX58i7SpU?usp=sharing)  
    - Training images  : 3000  
    - Validation images: 81  
    - Testing images   : 4481  

- Sample numbers of the training data with 8 classes:  

    |   Classes     |  Car   | Van|Truck|Walker|Sitter|Rider|Tram|Misc.|
    | ------------- |:------:|:--:|:---:|:----:|:----:|:---:|:--:|:---:|
    | Sample number |   11379|1197|  428|  1816|    93|  671| 208|  428|
    

## 2. Testing Result:
- ### YOLOv3
  - Mean average precision of each class:  
    |   Classes         |Car|  Van|Truck|Walker|Sitter|Rider|  Tram|Misc.|
    | ----------------- |:-:|:---:|:---:|:----:|:----:|:---:|:----:|:---:|
    | Average Precision |73%|35.9%|56.5%| 31.5%|    0%|30.5%| 33.8%| 0.3%|  
    
  - Performance:


- ### YOLOv4
- ### Scaled YOLOv4
- ### SSD (VGG-300)
- ### Faster RCNN
- ### Mask RCNN
