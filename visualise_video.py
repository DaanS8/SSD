import torch
from ssd.data import get_test_dataset, get_test_dataloader
from ssd.model import SSD300, ResNet
from ssd.train import load_checkpoint
from ssd.utils import dboxes300_coco, Encoder, SSDTransformer
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
import cv2
import os

from PIL import Image
writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 29.97, (854, 480))

path_model = 'save_models/epoch_64.pt'


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

if __name__ == '__main__':
    with torch.no_grad():
        args = dotdict({'larger_features': True,
                        'scales': [10, 21, 45, 99, 153, 207, 261],
                        'aspect_ratio_0': [2, ],
                        'aspect_ratio_1': [2, 3],
                        'aspect_ratio_2': [2, 3],
                        'aspect_ratio_3': [2, 3],
                        'aspect_ratio_4': [2, ],
                        'aspect_ratio_5': [2, ],
                        'nb_classes': 4,
        })

        dboxes = dboxes300_coco(args)
        test_trans = SSDTransformer(dboxes, (300, 300), val=True)

        ssd300 = SSD300(args, backbone=ResNet(args, 'resnet50'))
        ssd300.cuda()

        if os.path.isfile(path_model):
            load_checkpoint(ssd300, path_model)
            checkpoint = torch.load(path_model,
                                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']

        ssd300.eval()

        encoder = Encoder(dboxes)

        N_gpu = 1
        ret = []

        img_size = (480, 854)
        for file in sorted(os.listdir('2-4/')):
            print(file)
            image = Image.open('2-4/' + file)
            plt.imshow(image.convert('RGB'))
            img = test_trans.img_trans(image)
            # tensor_image = tensor_image.view(tensor_image.shape[1], tensor_image.shape[2], tensor_image.shape[0])
            img_batch = torch.empty(1, 3, 300, 300, dtype=torch.float32)
            img_batch[0] = img
            inp = img_batch.cuda()
            with torch.cuda.amp.autocast(enabled=True):
                # Get predictions
                ploc, plabel = ssd300(inp)

            ploc, plabel = ploc.float(), plabel.float()

            for idx in range(ploc.shape[0]):
                # ease-of-use for specific predictions
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)

                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, 0.50, 200)[0]
                except:
                    # raise
                    print("")
                    print("No object detected in idx: {}".format(idx))
                    continue

                htot, wtot = img_size[0], img_size[1]
                loc, label, prob = [r.cpu().numpy() for r in result]
                count, count_wrong = 0, 0
                best_ball = ([0, 0, 0, 0], 0, 0)
                for loc_, label_, prob_ in zip(loc, label, prob):
                    if label_ == 1 and prob_ > best_ball[2]:
                        best_ball = (loc_, label_, prob_)

                    if label_ == 3 and prob_ > 0.25:
                        count += 1
                        print(prob_, label_, (loc_[0] * wtot, loc_[1] * htot), (loc_[2] - loc_[0]) * wtot, (loc_[3] - loc_[1]) * htot)
                        color = ['r', 'g', 'b', 'y'][label_]
                        plt.gca().add_patch(Rectangle((loc_[0] * wtot, loc_[1] * htot), (loc_[2] - loc_[0]) * wtot, (loc_[3] - loc_[1]) * htot, linewidth=1, edgecolor=color, facecolor='none'))
                        # ret.append([img_id[idx], loc_[0] * wtot,
                        #             loc_[1] * htot,
                        #             (loc_[2] - loc_[0]) * wtot,
                        #             (loc_[3] - loc_[1]) * htot,
                        #             prob_,
                        #             inv_map[label_]])
                    else:
                        count_wrong += 1
                if best_ball[2] > 0.05:
                    print(f'prob ball: {best_ball[2]}')
                    color = ['r', 'g', 'b', 'y'][best_ball[1]]
                    loc_ = best_ball[0]
                    plt.gca().add_patch(
                        Rectangle((loc_[0] * wtot, loc_[1] * htot), (loc_[2] - loc_[0]) * wtot, (loc_[3] - loc_[1]) * htot,
                                  linewidth=1, edgecolor=color, facecolor='none'))

                print(count+1, count_wrong-1)
            plt.savefig('out-vid/' + file)
            plt.clf()