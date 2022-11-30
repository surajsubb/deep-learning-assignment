import argparse
import os
from solver import Enhancenet


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--Logger_folder', type=str, default="runs/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch96.pth")
	parser.add_argument('--enhanced_dir', type=str, default= "Dataset/enhanced")
	parser.add_argument('--test', type=bool, default= False)	
	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)
	if not os.path.exists(config.Logger_folder):
		os.mkdir(config.Logger_folder)
	if not os.path.exists(config.enhanced_dir):
		os.mkdir(config.enhanced_dir)
  
	model=Enhancenet(config)
	if model.test:
		model.Test()
	else:
		model.train()