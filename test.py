import sys

sys.path.append("ME-GraphAU/OpenGraphAU")
from model.MEFL import MEFARG

import yaml
import cv2
from utils import load_state_dict, image_eval, hybrid_prediction_infolist
from dataset import pil_loader
import torch
import torch.nn as nn

with open('conf.yaml', 'r') as file:
    conf = yaml.safe_load(file)

# load an image

# # run ME-Graph on it
class AU():
    def __init__(self):
        net = MEFARG(num_main_classes=conf["opengraphau"]["num_main_classes"], 
                    num_sub_classes=conf["opengraphau"]["num_sub_classes"], 
                    backbone=conf["opengraphau"]["backbone"])
        net = load_state_dict(net, conf["opengraphau"]["checkpoint"])

        net.eval()

        if torch.cuda.is_available():
            net = net.cuda()

        self.net = net
        self.img_transform = image_eval()
    
    def predict(self, img):
        img_ = self.img_transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            img_ = img_.cuda()
        with torch.no_grad():
            pred = self.net(img_)
            pred = pred.squeeze().cpu().numpy()
        return pred
    
    def draw_text(img, words, probs):
        AU_names = ['Inner brow raiser',
            'Outer brow raiser',
            'Brow lowerer',
            'Upper lid raiser',
            'Cheek raiser',
            'Lid tightener',
            'Nose wrinkler',
            'Upper lip raiser',
            'Nasolabial deepener',
            'Lip corner puller',
            'Sharp lip puller',
            'Dimpler',
            'Lip corner depressor',
            'Lower lip depressor',
            'Chin raiser',
            'Lip pucker',
            'Tongue show',
            'Lip stretcher',
            'Lip funneler',
            'Lip tightener',
            'Lip pressor',
            'Lips part',
            'Jaw drop',
            'Mouth stretch',
            'Lip bite',
            'Nostril dilator',
            'Nostril compressor',
            'Left Inner brow raiser',
            'Right Inner brow raiser',
            'Left Outer brow raiser',
            'Right Outer brow raiser',
            'Left Brow lowerer',
            'Right Brow lowerer',
            'Left Cheek raiser',
            'Right Cheek raiser',
            'Left Upper lip raiser',
            'Right Upper lip raiser',
            'Left Nasolabial deepener',
            'Right Nasolabial deepener',
            'Left Dimpler',
            'Right Dimpler']
        AU_ids = ['1', '2', '4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '22',
            '23', '24', '25', '26', '27', '32', '38', '39', 'L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6', 'L10', 'R10', 'L12', 'R12', 'L14', 'R14']
        # from PIL import Image, ImageDraw, ImageFont
        pos_y = img.shape[0] // 40
        pos_x  = img.shape[1] + img.shape[1] // 100
        pos_x_ =  img.shape[1]  * 3 // 2 - img.shape[1] // 100

        img = cv2.copyMakeBorder(img, 0,0,0,img.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255))
        # num_aus = len(words)
        # for i, item in enumerate(words):
        #     y = pos_y + (i * img.shape[0] // 17 )
        #     img = cv2.putText(img, str(item), (pos_x, y), cv2.FONT_HERSHEY_SIMPLEX, round(img.shape[1] / 2048, 3), (0,0,255), 2)
        # pos_y = pos_y + (num_aus * img.shape[0] // 17 )
        for i, item in enumerate(range(21)):
            y = pos_y  + (i * img.shape[0] // 22)
            color = (0,0,0)
            if float(probs[item]) > 0.5:
                color = (0,0,255)
            img = cv2.putText(img,  AU_names[i] + ' -- AU' +AU_ids[i] +': {:.2f}'.format(probs[i]), (pos_x, y), cv2.FONT_HERSHEY_SIMPLEX, round(img.shape[1] / 2800, 3), color, 2)

        for i, item in enumerate(range(21,41)):
            y = pos_y  + (i * img.shape[0] // 22)
            color = (0,0,0)
            if float(probs[item]) > 0.5:
                color = (0,0,255)
            img = cv2.putText(img,  AU_names[item] + ' -- AU' +AU_ids[item] +': {:.2f}'.format(probs[item]), (pos_x_, y), cv2.FONT_HERSHEY_SIMPLEX, round(img.shape[1] / 2800, 3), color, 2)
        return img


    def visualize(self, cv2_img, pred):
        infostr = {'AU prediction:'}
        infostr_probs,  infostr_aus = hybrid_prediction_infolist(pred, 0.5)
        infostr_aus = list(infostr_aus)
   
        img = self.draw_text(img, list(infostr_aus), pred)
        path = conf.input.split('.')[0]+'_pred.jpg'
        cv2.imwrite(path, img)
