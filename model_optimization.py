#### MODEL OPTIMIZATION TOOL ########
# by George Gorospe, george.gorospe@nmaia.net

# This tool optimizes the given pytorch model via the torch2trt library by Nvidia

# REQUIRED INPUTS: file path for .pth file to be optimized


# Importing required libraries
### Machine Learning Libraries
import torch
import torchvision
from torchvision.models import ResNet18_Weights




# ARGPARSE MODEL;
import argparse, os


device = torch.device('cuda')

# Resnet 18
model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(512, 2)
# Model Output
output_dim = 2
#model.to(device)

# Warm starting the new model to be optimized - loading weights from trained model into untrained model
#model = model_structure # this is the shape of the model before training
#model = model.cuda().eval().half()


# When executed, we should see, "<All keys matched successfully>" 

parser = argparse.ArgumentParser()
parser.add_argument("file_path", type=str,
                    help="optimize model at given location")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
args = parser.parse_args()
file_path = args.file_path

if os.path.isfile(file_path):
    print("Selected file: " + file_path)
    try:
        #model = model.cuda().eval().half()
        #model.load_state_dict(torch.load(model_file_path))
        model = torch.load(model_file_path)
        #model.eval()
        print("Model loaded successfully.")
    except:
        print("An exception occured while loading the model.")
else:
    print("No such file")


# TODO: complete model optimization code based on last two cells of 3_Train_Model.ipynb