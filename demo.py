from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import utils
import warnings
from utils.Functions import *
import utils.lib.pytorch_ssim as pytorch_ssim
import imageio
from skimage import metrics 
import numpy as np
import scipy.io as scio 
import scipy.misc as scim
import os,time
import logging,argparse
from datetime import datetime
from collections import OrderedDict
from einops import rearrange
from utils.LFDataset import LoadTestData
from utils.DeviceParameters import to_device

from model.HLFRN import HLFRN
from model.MSP import MSP
from model.DRLF import DRLF
from model.PFE import PFE



# Testing settings
parser = argparse.ArgumentParser(description="Light Field Restoration")
parser.add_argument("--model_name", type=str, default='HLFRN', help="Path for saving training log ")
parser.add_argument("--sigma", type=int, default=20, help="The number of stages")
parser.add_argument("--angResolution", type=int, default=5, help="The angular resolution of original LF")

parser.add_argument("--batchSize", type=int, default=1, help="Batch size")
parser.add_argument("--cropPatchSize", type=int, default=32, help="The size of croped LF patch")
parser.add_argument("--overlap", type=int, default=4, help="The size of croped LF patch")

parser.add_argument("--modelPath", type=str, default='./pretrained_models/HLFRN/model_sigma_10.pth', help="Path for loading trained model ")
parser.add_argument("--dataPath", type=str, default='./data/', help="Path for loading testing data ")
parser.add_argument("--savePath", type=str, default='./results/demo_real_img', help="Path for saving results ")
parser.add_argument("--cropImage", type=bool, default=True, help="Crop image to save memory during inference")


#  HLFRN parameters
parser.add_argument("--n_groups", type=int, default=5, help="The number of HGAG groups")
parser.add_argument("--n_blocks", type=int, default=5, help="The number of HFEB blocks")
parser.add_argument("--n_channels", type=int, default=32, help="The number of convolution filters")

#  DRLF parameters
parser.add_argument("--stageNum", type=int, default=3, help="The number of stages")
parser.add_argument("--channelNum", type=int, default=3, help="The number of input channels")

# PFE parameters
parser.add_argument("--temperature_1", type=float, default=1, help="The number of temperature_1")
parser.add_argument("--temperature_2", type=float, default=1, help="The number of temperature_2")
parser.add_argument("--component_num", type=int, default=4, help="The number of pfe component")
parser.add_argument("--sasLayerNum", type=int, default=6, help="The number of stages")
parser.add_argument("--epochNum", type=int, default=10000, help="The number of epoches")

opt = parser.parse_args()

save_dir = opt.savePath + '/' + opt.model_name + '_' + str(opt.sigma)
if not os.path.exists(save_dir): 
		os.makedirs(save_dir) 

# warnings.filterwarnings("ignore")
# plt.ion()
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# log = logging.getLogger()
# fh = logging.FileHandler( save_dir +  '/Testing_' + opt.model_name + '_' + str(opt.sigma) + '.log')
# log.addHandler(fh)

# logging.info(opt)

if __name__ == '__main__':

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if opt.model_name == 'DRLF':
		model= DRLF(opt)

	if opt.model_name == 'MSP':
		model= MSP(opt)
	
	if opt.model_name == 'PFE':
		model= PFE(opt)

	if opt.model_name == 'HLFRN':
		model=HLFRN(opt)


	model.load_state_dict(torch.load(opt.modelPath, map_location='cuda:0'))
	
	model.eval()
	to_device(model,device)

	total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	# log.info("Training parameters: %d" %total_trainable_params)

	scene_list = os.listdir(opt.dataPath)

	for scenes in scene_list:
		print('Working on scene: ' + scenes + '...')
		temp = imageio.imread(opt.dataPath + '/' + scenes + '/05_05.png')
		angRes = 9
		noiLF = np.zeros(shape=(angRes, angRes, temp.shape[0], temp.shape[1], 3)).astype('float32')

		for i in range(angRes):
			for j in range(angRes):
				name  = "%.2d_%.2d" % (i+1, j+1)
				# print(name)
				noiLF[i, j, :, :, :] = imageio.imread(opt.dataPath + '/' + scenes + '/' + name + '.png')

		noiLF = torch.from_numpy(noiLF) / 255.0
		noiLF =  noiLF.unsqueeze(0)
		b, u,v,h,w,c = noiLF.shape
		noiLF = noiLF[:,(u-opt.angResolution)//2:(u+opt.angResolution)//2, (v-opt.angResolution)//2:(v+opt.angResolution)//2, :, :, 0:3]
		# print(noiLF.shape)
		with torch.no_grad(): 
			if opt.cropImage:
				cropStride= opt.cropPatchSize-opt.overlap
				noiLFStack,coordinate=CropLF(noiLF,opt.cropPatchSize, cropStride) #[b,n,u,v,x,y,c]
				b,n,u,v,x,y,c=noiLFStack.shape
				denoilfStack=torch.zeros(b,n,u,v,x,y,c)#[b,n,u,v,x,y,c]
								
				# reconstruction
				avg_time_patch = 0
				for i in range(noiLFStack.shape[1]):
					if opt.model_name == 'MSP':
						_,_,denoiLFPatch=model(noiLFStack[:,i,:,:,:,:].permute(0,1,2,5,3,4).cuda())  #[b,u,v,c,x,y]
					else:
						if opt.model_name == 'PFE':
							epoch = 10000
							denoiLFPatch=model(noiLFStack[:,i,:,:,:,:].permute(0,1,2,5,3,4).cuda(),epoch)  #[b,u,v,c,x,y]
						else:
							denoiLFPatch=model(noiLFStack[:,i,:,:,:,:].permute(0,1,2,5,3,4).cuda())  #[b,u,v,c,x,y]

					denoilfStack[:,i,:,:,:,:,:]= denoiLFPatch.permute(0,1,2,4,5,3) #[b,n,u,v,x,y,c]

				denoiLF=MergeLF(denoilfStack,coordinate,opt.overlap) #[b,u,v,x,y,c]
				b,u,v,x,y,c=denoiLF.shape   
			

			else:
				denoiLF=model(noiLF.permute(0,1,2,5,3,4).cuda())  #[b,u,v,c,x,y]
				denoiLF = denoiLF.permute(0,1,2,4,5,3)
				b,u,v,x,y,c=denoiLF.shape

		save_png_dir = os.path.join(save_dir,scenes)
		if not os.path.exists(save_png_dir): 
			os.makedirs(save_png_dir) 

		# # ''' Save RGB '''
		if save_png_dir is not None:
			denoiLF = denoiLF.squeeze(0)
			denoiLF = 255 * denoiLF.cpu().numpy()
			denoiLF = np.clip(denoiLF, 0, 255)
			
			# save all views
			for i in range(opt.angResolution):
				for j in range(opt.angResolution):
					img = np.uint8(denoiLF[i, j, :, :, :])
					path = str(save_png_dir) + '/' + 'View' + '_' + str(i) + '_' + str(j) + '.png'
					imageio.imwrite(path, img)
					pass
				pass
			pass
		
	print('Finish.')