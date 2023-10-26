# SiBNN
Code for papers:
* 'Sparsity-Inducing Binarized Neural Networks', AAAI 2020
* 'Toward Accurate Binarized Neural Networks With Sparsity for Mobile Application', TNNLS 2022.

SiBNN is a simple and effective binary quantization framework for CNN.


# Train:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main_sibnn_learn.py -a resnet18_sibnn_bireal_learn --data /PATH/TO/THE/DATA


# Results:
    \* Prec@1 62.166 Prec@5 83.682


# Related Papers

    Please cite our paper if it helps your research:

    @inproceedings{wang2020sparsity,
        title={Sparsity-inducing binarized neural networks},
        author={Wang, Peisong and He, Xiangyu and Li, Gang and Zhao, Tianli and Cheng, Jian},
        booktitle={Proceedings of the AAAI conference on artificial intelligence},
        volume={34},
        number={07},
        pages={12192--12199},
        year={2020}
    }

    @article{wang2022toward,
        title={Toward accurate binarized neural networks with sparsity for mobile application},
        author={Wang, Peisong and He, Xiangyu and Cheng, Jian},
        journal={IEEE Transactions on Neural Networks and Learning Systems},
        year={2022},
        publisher={IEEE}
    }
