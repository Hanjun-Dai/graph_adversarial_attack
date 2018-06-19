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
    |    |__ ......
    |......
    
