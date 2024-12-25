# HLFRN
This is the code implementation of paper: "Hybrid Spatial and Frequency Network for Light Field Image Restoration"

### Dependencies and Installation
- Pytorch == 2.3.0
- CUDA == 12.1
```
# git clone this repository
git clone https://github.com/duongvinh/HLFRN.git
cd HLFRN
```

### Running Examples

### 1. Dataset
Please first download light field datasets in [here](https://drive.google.com/drive/folders/1oJFOQtf-OSARfC2vslJlw1CFr1OlQe4V?usp=drive_link). 


### 2. Test

___For sythesis test data___

We provide the pre-trained models for adding zero-mean Gaussian noise with the standard variance varying in the range of 10, 20, and 50 on the Lytro dataset. Enter the scripts folder and run:

```
python test.py \
    --model_name HLFRN \
    --sigma 50 \
    --modelPath ./pretrained_models/HLFRN/model_sigma_50.pth \
    --dataPath  ./test_noiseLeve_10-20-50_4-11_5x5.mat \
    --savePath ./results/sythesis_img_test/ \
```
___For real test data___

We have used Lytro Illum camera to capture various real-world LF image under different conditions. Please download via [Google Drive](https://drive.google.com/drive/folders/18kBOQuvAtTVDhWPnw5IJJAShv_M1m0zW)

```
python demo.py \
    --model_name HLFRN \
    --sigma 50 \
    --modelPath ./pretrained_models/HLFRN/model_sigma_50.pth \
    --dataPath ./data/ \
    --savePath ./results/real_img_test \
```

For test other methods, we just need to modify "--model_name" to MSP, DRLF, or PFE methods. More examples can be seen in "./scripts" folder.

### 3. Train
Enter the scripts folder and run:

```
python train.py \
    --model_name HLFRN \
    --sigma 50 \
    --dataPath  ./train_noiseLevel_10-20-50_4-11_color_5x5.mat \
    --saveCheckpointsDir ./checkpoints/ \
```

For train other methods, we just need to modify "--model_name" to MSP, DRLF, or PFE methods. More examples can be seen in "./scripts" folder.

### Citation
If our work is useful for your research, please consider citing:

```Citation
@Article{vinh2023-lfsr,
  author  = {Duong, V. V. and Nguyen, T. H. and Yim, J. and Jeon, B.},
  journal = {submitted IEEE Trans. Compuational Imaging},
  title   = {Hybrid Spatial and Frequency Network for Light Field Image Restoration},
  year    = {2024},
}
```

### Acknowledgement

We would like to thanks the authors of  [DRLF](https://github.com/MantangGuo/DRLF/tree/main), [PFE](https://github.com/lyuxianqiang/LFCA-CR-NET), and [MSP](https://github.com/shuozh/MSP) for sharing code.

### Contact
If you have any questions, please feel free to reach me out at `duongvinh@skku.edu`.