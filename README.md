# graph_adversarial_attack
Adversarial Attack on Graph Structured Data (https://arxiv.org/abs/1806.02371, to appear in ICML 2018). 
This repo contains the code, data and results reported in the paper.

### 1. download repo and data

First clone the repo recursively, since it depends on another repo (https://github.com/Hanjun-Dai/pytorch_structure2vec):

    git clone git@github.com:Hanjun-Dai/graph_adversarial_attack --recursive

(BTW if you have trouble downloading it because of permission issues, please see [this issue](https://github.com/Hanjun-Dai/graph_adversarial_attack/issues/2) )

Then download the data using the following dropbox link:

    https://www.dropbox.com/sh/mu8odkd36x54rl3/AABg8ABiMqwcMEM5qKIY97nla?dl=0

Put everything under the 'dropbox' folder, or create a symbolic link with name 'dropbox':

    ln -s /path/to/your/downloaded/files dropbox
    
Finally the folder structure should look like this:

    graph_adversarial_attack (project root)
    |__  README.md
    |__  code
    |__  pytorch_structure2vec
    |__  dropbox
    |__  |__ data
    |    |__ scratch
    |......
    
Optionally, you can use the data generator under ``code/data_generator`` to generate the synthetic data.

### 2. install dependencies and build c++ backend

The current code depends on pytorch 0.3.1, cffi and CUDA 9.1. Please install using the following command (for linux):

    pip install http://download.pytorch.org/whl/cu91/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl 
    pip install cffi==1.11.2
    
The c++ code needs to be built first:

    cd pytorch_structure2vec/s2v_lib
    make
    cd code/common
    make

### 3. Train the graph classification and node classification model (our attack target)

If you want to retrain the target model, go to either ``code/graph_classification`` or ``code/node_classification`` and run the script in train mode. For example:

    cd code/graph_classification
    ./run_er_components.sh -phase train

You can also use the pretrained model that is the same as used in this paper, under the folder ``dropbox/scratch/results``

### 4. Attack the above trained model. 

In this paper, we presented 5 different approaches for attack, under both graph-level classification and node-level classification tasks. The code for attack can be found under ``code/graph_attack`` and ``code/node_attack``, respectively. 

For example, to use Q-leaning to attack the graph classification method, do the following:

    cd code/graph_attack
    ./run_dqn.sh -phase train

### Reference 

    @article{dai2018adversarial,
      title={Adversarial Attack on Graph Structured Data},
      author={Dai, Hanjun and Li, Hui and Tian, Tian and Huang, Xin and Wang, Lin and Zhu, Jun and Song, Le},
      journal={arXiv preprint arXiv:1806.02371},
      year={2018}
    }


