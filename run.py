import argparse
from modules.model import Model


parser = argparse.ArgumentParser(description='train covid-diagnosis')
parser.add_argument('--model_name', required=True,
                    help='choose model name')
parser.add_argument('--backbone', required=True,
                    help='choose backbone for network')
parser.add_argument('--dataset', required=True,
                    help='choose dataset from x-ray & CT scan data')
parser.add_argument('--grad_cam', default=False,
                    help='visualization of heat map')



args = parser.parse_args()

test_model = Model(args.model_name, args.backbone)
test_model.set_dataset(args.dataset)
test_model.train()






