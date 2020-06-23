# Volumetric Heatmap Autoencoder
### Accepted to CVPR 2020

This repo contains the code related to the paper [Compressed Volumetric Heatmaps for Multi-Person 3D Pose Estimation](https://arxiv.org/abs/2004.00329) accepted to CVPR 2020 with the instructions for training the Volumetri Heatmap Autencoder on JTA dataset. [Here](https://github.com/fabbrimatteo/LoCO)
you can also find the code for training the full pipeline.

## Intructions
- Download the [JTA dataset](http://aimagelab.ing.unimore.it/jta)
 in `<your_jta_path>`
- Run `python to_poses.py --out_dir_path='poses' --format='torch'` 
([link](https://github.com/fabbrimatteo/JTA-Dataset)) 
to generate the `<your_jta_path>/poses` directory
- Run `python to_imgs.py --out_dir_path='frames' --img_format='jpg'`
([link](https://github.com/fabbrimatteo/JTA-Dataset)) 
 to generate the `<your_jta_path>/frames` directory
- Modify the `conf/default.yaml` configuration file specifying the 
path to the JTA dataset directory
     - `JTA_PATH: <your_jta_path>`

#### Show Paper Results
- Modify the `conf/pretrained.yaml` configuration file specifying the path to the JTA dataset directory
     - `JTA_PATH: <your_jta_path>`
- run `python show.py pretrained` (python >= 3.6)

#### Train
- run `python main.py default` (python >= 3.6)

TIP: training using sparse ground truth is not trivial since the network
will output maps with only zeros no matter what. 
A practical way to overcome this problem is to start the training 
with sigma 8 and then halve the sigma whenever you encounter a loss plateaus.
In our experiments we stopped when fully trained at sigma 2.


## Citation

We believe in open research and we are happy if you find this data useful.   
If you use it, please cite our work.

```latex
@inproceedings{fabbri2020compressed,
   title     = {Compressed Volumetric Heatmaps for Multi-Person 3D Pose Estimation},
   author    = {Fabbri, Matteo and Lanzi, Fabio and Calderara, Simone and Alletto, Stefano and Cucchiara, Rita},
   booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
   year      = {2020}
 }
```
