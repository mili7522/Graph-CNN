# Graph-CNN

## Note for COMP5329 Assignment 2
All code to preprocess data is present in the 'Data' folder. The GIS shapefiles need to be downloaded from the ABS (http://www.abs.gov.au/AUSSTATS/abs@.nsf/DetailsPage/1270.0.55.001July%202016?OpenDocument).
The files 'Statistical Area Level 1 (SA1) ASGS Ed 2016 Digital Boundaries in ESRI Shapefile Format' was used, containing all SA1 regions of Australia.

All test can be run from the command line by specifying the number of layer groups (consisting of an embedding layer and graph convolution layer), number of neurons per layer and the repetition number (0 to 4 for 5 fold repetition).
Default values of 2, 64 and 0 are used if not enough parameters are specified.
Required inputs are already present in the 'Data' folder.

-------
Implementation of "Robust Spatial Filtering with Graph Convolutional Neural Networks" - https://arxiv.org/abs/1703.00792.
This project is still under development. Email Felipe at fps7806@rit.edu if you have any suggestions or problems.
## Requirements:
* Tensorflow >1.0
* Internet Connection (if automatically downloading datasets)

## Instructions
To obtain results on NCI1 and Cora run the following commands:

`python src/run_nci1.py`

`python src/run_cora.py`

The scripts will automatically download the datasets and run the experiments.

Result are automatically saved in the results folder. `python src/parse_results.py results/Cora.txt` will parse the results and display useful information.

Note that the results will not be exactly the same every run.

Scripts will print useful information. Can be disabled by adding `exp.silent = True`.

Checkpoints and Summaries that can be used with Tensorboard will be generated if the line `self.debug = True` is removed.

There is an example on how to run experiments on ImageNet under `src/run_alexnet.py`. The training was not optimized so the final accuracy might not match the expected.

## NCI1 - Results

| Model        | File | Accuracy           |
| ------------- |:------------- |:-------------:|
| 64-64-GEP(32)-32-GEP(8)-FC256-FC2 |   run_nci1.py   | 84.45 +- 0.94 |

## Cora - Results

| Model        | File | Accuracy           |
| ------------- |:------------- |:-------------:|
| Embed256-48-Embed32-48-Embed32-7 |   run_cora.py   | 89.18 +- 1.96 |

## Citation

Please cite us in your publications if it helps your research:

    @article{graphcnn2017,
      Author = {Petroski Such, Felipe, and Sah, Shagan, and Dominguez, Miguel, and Pillai, Suhas, and Zhang, Chao, and Michael,  Andrew, and Cahill, Nathan, and Ptucha, Raymond},
      Journal = {arXiv preprint arXiv:1703.00792},
      Title = {Robust Spatial Filtering with Graph Convolutional Neural Networks},
      Year = {2017}
    }