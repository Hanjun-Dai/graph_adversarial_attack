# graph_adversarial_attack
Adversarial Attack on Graph Structured Data (https://arxiv.org/abs/1806.02371)

# 1. download repo and data

First clone the repo recursively, since it depends on another repo (https://github.com/Hanjun-Dai/pytorch_structure2vec):

    git clone git@github.com:Hanjun-Dai/graph_adversarial_attack --recursive

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

# 2. install dependencies and build c++ backend

The current code depends on pytorch 0.3.1, cffi and CUDA 9.1. Please install using the following command (for linux):

    pip install http://download.pytorch.org/whl/cu91/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl 
    pip install cffi
    
The c++ code needs to be built first:

    cd pytorch_structure2vec/s2v_lib
    make
    cd code/common
    make

# 3. Train the graph classification and node classification model (our attack target)

If you want to retrain the target model, go to either ``code/graph_classification`` or ``code/node_classification`` and run the script in train mode. For example:

    cd code/graph_classification
    ./run_er_components.sh -phase train

You can also use the pretrained model that is the same as used in this paper, under the folder ``dropbox/scratch/results``

    
    
