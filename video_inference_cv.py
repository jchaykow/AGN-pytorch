import time
import cv2
import numpy as np
import glob
import os
import dlib
import argparse
from scipy import misc
from itertools import cycle

import torch
from torch.autograd import Variable

import pygame, sys
import pygame.locals

pygame.init()

from imports import *
from archs import *

from models.inception_resnet_v1 import InceptionResnetV1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
# Load Model FT pretrained
class_names = [
    'Adrien_Brody','Alejandro_Toledo','Angelina_Jolie','Arnold_Schwarzenegger','Carlos_Moya',
    'Charles_Moose','James_Blake','Jennifer_Lopez','Michael_Chaykowsky','Roh_Moo-hyun','Venus_Williams']
model_ft = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes=len(class_names))
layer_list = list(model_ft.children())[-5:]
model_ft = nn.Sequential(*list(model_ft.children())[:-5])
model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
model_ft.last_linear = nn.Sequential(
    Flatten(),
    nn.Linear(in_features=1792, out_features=512, bias=False),
    normalize()
)
model_ft.logits = nn.Linear(layer_list[3].in_features, len(class_names))
model_ft.softmax = nn.Softmax(dim=1)
model_ft = model_ft.to(device)

# Build masks
imgGlass = cv2.imread("data/glasses_mask.png", -1)
r = 160.0 / imgGlass.shape[1]
dim = (160, int(imgGlass.shape[0] * r))
imgGlass = cv2.resize(imgGlass, dim, interpolation = cv2.INTER_AREA)
imgGlass = imgGlass[39:81, 21:138]
alpha_data = imgGlass[:,:,0] + imgGlass[:,:,1] + imgGlass[:,:,2]
alpha_data[alpha_data < 200] = 0
alpha_data[alpha_data > 20] = 255
orig_mask_g = alpha_data
orig_mask_inv_g = cv2.bitwise_not(orig_mask_g)
print(orig_mask_inv_g.shape)

bs,sz,nz = 64,64,100
nc = 3; ndf = 160; ngf = 160

predictor_path = "data/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "data/dlib_face_recognition_resnet_model_v1.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1("data/mmod_human_face_detector.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')

unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
renorm = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

def main(file, output, frame_rate=30):
    netGa = Generator(1).to(device)
    netGb = Generator(1).to(device)

    # Load models
    if torch.cuda.is_available():
        generator_A = torch.load(args.modelA).cuda()
        generator_B = torch.load(args.modelA).cuda()
    # Since the models were trained on GPU we have to remap them to CPU
    else:
        checkpointA = torch.load(
            args.modelA, map_location=lambda storage, loc: storage)
        netGa.load_state_dict(checkpointA['state_dict'])
        checkpointB = torch.load(
            args.modelB, map_location=lambda storage, loc: storage)
        netGb.load_state_dict(checkpointB['state_dict'])
        checkpointC = torch.load(
            args.modelft, map_location=lambda storage, loc: storage)
        model_ft.load_state_dict(checkpointC['state_dict'])
    
    # Switch for generator
    generator = cycle([0,1,2])
    gen = next(generator)
    generators = [netGa, netGb]

    fixed_noise = torch.randn(1, nz, 1, 1, device=device)
    fake = generators[gen](fixed_noise).cpu()
    fakes = unorm_glasses(fake)
    fakes = np.transpose(to_np(fakes[:,:,39:81,21:138]), (0,2,3,1))
    for k in range(fakes.shape[3]):
        fakes[0,:,:,k][orig_mask_g == 0] = 0
    fakes = (fakes[0] * 255).astype('uint8')
    origGlassHeight, origGlassWidth = fakes.shape[:2]

    if (file == "camera"):
        video_capture = cv2.VideoCapture(0)
    else:
        video_capture = cv2.VideoCapture(file)
    # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 160)

    ret, frame = video_capture.read()
    r = 400.0 / frame.shape[1]
    dim = (400, int(frame.shape[0] * r))
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    if (output != None):
        out = cv2.VideoWriter(output, fourcc, frame_rate, (frame.shape[1], frame.shape[0]))
    
    while ret:
        dets = cnn_face_detector(frame, 1)
        if not dets: continue
        for event in pygame.event.get():
            if event.type == pygame.locals.KEYDOWN:
                if event.key == pygame.K_p:
                    gen = next(generator)
                    if gen == 2: break
                    fixed_noise = torch.randn(1, nz, 1, 1, device=device)
                    fake = generators[gen](fixed_noise).cpu()
                    fakes = unorm_glasses(fake)
                    fakes = np.transpose(to_np(fakes[:,:,39:81,21:138]), (0,2,3,1))
                    for k in range(fakes.shape[3]):
                        fakes[0,:,:,k][orig_mask_g == 0] = 0
                    fakes = (fakes[0] * 255).astype('uint8')
                    origGlassHeight, origGlassWidth = fakes.shape[:2]
                elif event.key == pygame.K_q:
                    break
        for k, d in enumerate(dets):
            if gen != 2: 
                #set_trace()
                shapes = predictor(frame, d.rect)

                print('generator: ', gen)

                glassWidth = abs(shapes.part(16).x - shapes.part(1).x)
                glassHeight = int(glassWidth * origGlassHeight / origGlassWidth)
                
                y1 = int(shapes.part(24).y)
                y2 = int(y1 + glassHeight)
                x1 = int(shapes.part(27).x - (glassWidth/2))
                x2 = int(x1 + glassWidth)
                if y1 < 0: 
                    glassHeight = glassHeight - abs(y1)
                    y1 = 0
                if y2 > frame.shape[0]: glassHeight = glassHeight - (y2 - frame.shape[0])
                if x1 < 0: 
                    glassWidth = glassWidth - abs(x1)
                    x1 = 0
                if x2 > frame.shape[1]: glassWidth = glassWidth - (x2 - frame.shape[1])
                
                glass = cv2.resize(fakes, (glassWidth,glassHeight), interpolation = cv2.INTER_AREA)
                mask = cv2.resize(orig_mask_g, (glassWidth,glassHeight), interpolation = cv2.INTER_AREA)
                mask = np.stack((mask,)*3, axis=-1)
                mask_inv = cv2.resize(orig_mask_inv_g, (glassWidth,glassHeight), interpolation = cv2.INTER_AREA)
                mask_inv = np.stack((mask_inv,)*3, axis=-1)

                roi1 = frame[y1:y2, x1:x2]
                roi_bg = cv2.bitwise_and(roi1,mask_inv)
                roi_fg = cv2.bitwise_and(glass,mask)
                frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)

            model_ft.eval()
            frame_ft = frame[np.maximum(d.rect.top()-10, 0):np.minimum(d.rect.bottom()+12, frame.shape[0]), np.maximum(d.rect.left()-2, 0):np.minimum(d.rect.right()+6, frame.shape[1]),:]
            frame_ft = misc.imresize(frame_ft, (160,160), interp='bilinear')
            # frame = cv2.resize(frame, (160,160), interpolation = cv2.INTER_AREA)
            #frame_ft = cv2.cvtColor(frame_ft, cv2.COLOR_BGR2RGB)
            frame_ft_T = renorm(T(np.transpose(frame_ft.astype(np.int64) / 255, (2,0,1))[None,:,:,:])).to(device)
            outputs = model_ft(frame_ft_T)
            print(outputs)
            _, pred = torch.max(outputs, 1)
            pred_text = class_names[pred]
            print(pred_text)

            cv2.rectangle(frame,(np.maximum(d.rect.left()-2, 0),np.maximum(d.rect.top()-10, 0)),(np.minimum(d.rect.right()+6, frame.shape[1]), np.minimum(d.rect.bottom()+12, frame.shape[0])),(0,201,84),2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,f'{pred_text}',(d.rect.left(),np.maximum(d.rect.top()-20,0)), font, 0.3, (240,240,240), 1, cv2.LINE_AA)

        if (output != None):
            out.write(frame)
        else:
            cv2.imshow("", frame)
        ret, frame = video_capture.read()
        # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 160)
        r = 400.0 / frame.shape[1]
        dim = (400, int(frame.shape[0] * r))
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # if cv2.waitKey(1) & 0xFF == ord('s'):
        #     generator ^= 1
        # Release handle to the webcam
    if (output != None):
        out.release()
    video_capture.release()
    cv2.destroyAllWindows()


def unorm_glasses(fake):
    return (fake - fake.min())/(fake.max() - fake.min())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--modelA',
        type=str,
        default=os.path.join('data', 'tmp', 'G_20epochs_09132019.pth.tar'),
        help='Path to model file')
    parser.add_argument(
        '--modelB',
        type=str,
        default=os.path.join('data', 'tmp', 'AGN_G_09232019_Jennifer_Lopez_3.pth.tar'),
        help='Path to model file')
    parser.add_argument(
        '--modelft',
        type=str,
        default=os.path.join('data', 'tmp', 'model_ft_acc98_09202019.pth.tar'),
        help='Path to model file')
    parser.add_argument("-f", "--file", type=str, help="give video file for filter write camera if you want to use webcam", required=True)
    parser.add_argument("-o", "--output", type=str, help="give output name for video in .mp4 format")
    parser.add_argument("-fr", "--frame_rate", type=str, help="give video frame", default=10)
    args = parser.parse_args()

    file = args.file
    output = args.output
    frame_rate = args.frame_rate
    main(file, output, frame_rate)