# Structural RNN using PyTorch

This code/implementation is available for research purposes. If you are using this code for your work, please cite the following paper

> Anirudh Vemula, Katharina Muelling and Jean Oh. **Social Attention : Modeling Attention in Human Crowds**. *Submitted to the International Conference on Robotics and Automation (ICRA) 2018*.

Or use the following BibTeX entry
```

@ARTICLE{2017arXiv171004689V,
   author = {{Vemula}, A. and {Muelling}, K. and {Oh}, J.},
    title = "{Social Attention: Modeling Attention in Human Crowds}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1710.04689},
 primaryClass = "cs.RO",
 keywords = {Computer Science - Robotics, Computer Science - Learning},
     year = 2017,
    month = oct,
   adsurl = {http://adsabs.harvard.edu/abs/2017arXiv171004689V},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

**Author** : Anirudh Vemula

**Affiliation** : Robotics Institute, Carnegie Mellon University

**License** : GPL v3

## Requirements
* Python 3
* Seaborn (https://seaborn.pydata.org/)
* PyTorch (http://pytorch.org/)
* Numpy
* Matplotlib
* Scipy

## How to Run
* Before running the code, create the required directories by running the script `make_directories.sh`
* To train the model run `python srnn/train.py` (See the code to understand all the arguments that can be given to the command)
* To test the model run `python srnn/sample.py --epoch=n` where `n` is the epoch at which you want to load the saved model. (See the code to understand all the arguments that can be given to the command)
