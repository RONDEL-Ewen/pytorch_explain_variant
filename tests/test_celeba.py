import sys, os
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the desired directory
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the desired directory to sys.path
sys.path.append(os.path.abspath('C:/Users/migue/Documents/Faculdade/8-Semestre-erasmus/XAI/pytorch_explain_variant'))
from torch_explain.datasets import celebA
from tests.test_dcr import train_concept_bottleneck_model

x, c, y = celebA(1000)
train_concept_bottleneck_model(x, c, y, embedding_size=16)
