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

    <img src="figures/v3_1.jpg" alt="arch" width="750" style="zoom:100%;" />  
    <img src="figures/v3_2.jpg" alt="arch" width="750" style="zoom:100%;" />  
    <img src="figures/v3_3.jpg" alt="arch" width="750" style="zoom:100%;" />  
    
- ### YOLOv4  

  - Mean average precision of each class:  
    |   Classes         |Car|  Van|Truck|Walker|Sitter|Rider|  Tram|Misc.|
    | ----------------- |:-:|:---:|:---:|:----:|:----:|:---:|:----:|:---:|
    | Average Precision |63.6%|63%|79%| 26.7%|   19.1%|40.1%| 57.5%| 50.2%|  
    
  - Performance:  

    <img src="figures/v4_1.jpg" alt="arch" width="750" style="zoom:100%;" />  
    <img src="figures/v4_2.jpg" alt="arch" width="750" style="zoom:100%;" />  
    <img src="figures/v4_3.jpg" alt="arch" width="750" style="zoom:100%;" />  

    
- ### Scaled YOLOv4  

  - Mean average precision of each class:  
    |   Classes         |Car|  Van|Truck|Walker|Sitter|Rider|  Tram|Misc.|
    | ----------------- |:-:|:---:|:---:|:----:|:----:|:---:|:----:|:---:|
    | Average Precision |65.7%|60.9%|65.3%| 23.5%|   15.0%|60.4%| 62.9%| 59.1%|  
    
  - Performance:  
    
    <img src="figures/sv4_1.jpg" alt="arch" width="750" style="zoom:100%;" />  
    <img src="figures/sv4_2.jpg" alt="arch" width="750" style="zoom:100%;" />  
    <img src="figures/sv4_3.jpg" alt="arch" width="750" style="zoom:100%;" />  
    
- ### SSD (VGG-300)
- ### Faster RCNN
- ### Mask RCNN
