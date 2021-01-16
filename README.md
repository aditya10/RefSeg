# RefSeg

Improving Refererring Image Segmentation based on the CMPC model.

## Setup

We recommended the following dependencies.

* Python 2.7
* TensorFlow 1.5
* Numpy
* pydensecrf

This code is derived from [RRN](https://github.com/liruiyu/referseg_rrn) \[2\]. Please refer to it for more details of setup.

## Data Preparation
* Dataset Preprocessing

We conduct experiments on 4 datasets of referring image segmentation, including `UNC`, `UNC+`, `Gref` and `ReferIt`. After downloading these datasets, you can run the following commands for data preparation:
```
python build_batches.py -d Gref -t train
python build_batches.py -d Gref -t val
python build_batches.py -d unc -t train
python build_batches.py -d unc -t val
python build_batches.py -d unc -t testA
python build_batches.py -d unc -t testB
python build_batches.py -d unc+ -t train
python build_batches.py -d unc+ -t val
python build_batches.py -d unc+ -t testA
python build_batches.py -d unc+ -t testB
python build_batches.py -d referit -t trainval
python build_batches.py -d referit -t test
```

* Glove Embedding

Download `Gref_emb.npy` and `referit_emb.npy` and put them in `data/`. We provide download link for Glove Embedding here:
[Baidu Drive](https://pan.baidu.com/s/19f8CxT3lc_UyjCIIE_74FA), password: 2m28.


## Training
Train on UNC training set with:
```
python -u trainval_model.py -m train -d unc -t train -n CMPC_model -emb -f ckpts/unc/cmpc_model
```

## Testing
Test on UNC validation set with:
```
python -u trainval_model.py -m test -d unc -t val -n CMPC_model -i 700000 -c -emb -f ckpts/unc/cmpc_model
```

## Reference

CVPR 2020 paper: [*Referring Image Segmentation via Cross-Modal Progressive Comprehension*](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_Referring_Image_Segmentation_via_Cross-Modal_Progressive_Comprehension_CVPR_2020_paper.pdf).
Shaofei Huang*, Tianrui Hui*, Si Liu, Guanbin Li, Yunchao Wei, Jizhong Han, Luoqi Liu, Bo Li (* Equal contribution)

```
@inproceedings{huang2020referring,
  title={Referring Image Segmentation via Cross-Modal Progressive Comprehension},
  author={Huang, Shaofei and Hui, Tianrui and Liu, Si and Li, Guanbin and Wei, Yunchao and Han, Jizhong and Liu, Luoqi and Li, Bo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10488--10497},
  year={2020}
}
```
In addition:

\[1\] Chen, Ding-Jie, et al. "See-through-text grouping for referring image segmentation." Proceedings of the IEEE International Conference on Computer Vision. 2019.

\[2\] Li, Ruiyu, et al. "Referring image segmentation via recurrent refinement networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
