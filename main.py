
import cv2
import os
import argparse
import os.path as osp
from glob import glob

import numpy as np
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from sklearn.metrics import calinski_harabasz_score

from centerface import CenterFace


img_ext=['jpg','jpeg','png','bmp']


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-p", "--path", default='./image/', help="path to input directory of photos")
    args.add_argument("-f", "--face", default=True, action='store_true', help="is save face image")
    args = args.parse_args()

    if osp.exists(osp.join(args.path,'person.csv')) and osp.exists(osp.join(args.path,'cache.npy')):
        with open(osp.join(args.path,'person.csv'),'r') as f:
            exists = [x.split(',')[0] for x in f.readlines()]
        cache = np.load(osp.join(args.path,'cache.npy'),allow_pickle=True).tolist()
        embedding = cache['embedding'].tolist()
        person = cache['person']
        face_idx=len(embedding)
    else:
        face_idx=0
        embedding = []
        exists=[]
        person={}


    img_list = [x for x in glob(osp.join(args.path,'*')) if x.split('.')[-1].lower() in img_ext and osp.basename(x) not in exists]

    centerface = CenterFace(landmarks=True)
    embedder = cv2.dnn.readNetFromTorch('models/nn4.small2.v1.t7')

    
    for p in tqdm(img_list):
        person[osp.basename(p)]=[]
        img = cv2.imread(p)
        dets, lms = centerface(img, img.shape[0], img.shape[1], threshold=0.35)

        for det in dets:
            x1,y1,x2,y2 = [int(x) for x in det[:4]]
            face = img[y1:y2,x1:x2]
            face = cv2.resize(face,(128,128))
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward().flatten()
            embedding.append(vec)
            person[osp.basename(p)].append(face_idx)
            if args.face:
                if not osp.exists(osp.join(args.path,'face')) or not osp.isdir(osp.join(args.path,'face')):
                    os.makedirs(osp.join(args.path,'face'))
                cv2.imwrite(osp.join(args.path,'face','{}.png'.format(face_idx)),face)
            face_idx+=1
    
    embedding = np.array(embedding)
    best_score, best_k, best_gamma, best_pre = 0,0,0,[]
    for gamma in (0.01, 0.1, 1, 5):
        for k in (3, 4, 5, 6):
            y_pre = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(embedding)
            score = calinski_harabasz_score(embedding, y_pre)
            if score > best_score:
                best_score = score
                best_k = k 
                best_gamma = gamma
                best_pre = y_pre

    with open(osp.join(args.path,'person.csv'),'w') as f:
        for k,v in person.items():
            line=','.join([k]+[str(y_pre[x]) for x in v])
            f.write(line+'\n')
    np.save(osp.join(args.path,'cache.npy'),{'person':person,'embedding':embedding})
