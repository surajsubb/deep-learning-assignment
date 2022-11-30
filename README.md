# LOL FACE
Low Light Face detection Enhancement Network Inspired from U-Net and Mask-RCNN
# Instructions to execute 

1. Download the dataset from the link given in the “Dataset Used” section.
2. Install the requirements.
```python
 pip install -r requirements.txt
 ```
3. Create a folder called “Dataset” and put the data in the folder.
4. For training the image enhancements execute:
```python
 python main.py –-lowlight_images_path “./Datasets/image/” --num_epochs 200
 ```
5. For testing the image enhancements execute: 
```python
python main.py –-lowlight_images_path “./Datasets/image/”  --pretrain_dir “Snapshots/Epoch96.pth” –-test True –load_pretrain True
```
6. After executing the above commands, a enhanced folder will be created with the enhanced images.
7. Now execute the face_detection.ipnyb file to train the face detection model and get the detection from it.
8. For training the model without the enhancements execute the file “Without_enhanced_images.ipnyb”
