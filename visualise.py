import torch
from ssd.data import get_test_dataset, get_test_dataloader
from ssd.model import SSD300, ResNet
from ssd.train import load_checkpoint
from ssd.utils import dboxes300_coco, Encoder
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle

path_model = 'save_models/epoch_64.pt'
dataset = 'data/'
output_folder = 'output_5'
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
if __name__ == '__main__':
    with torch.no_grad():
        args = dotdict({"larger_features": True,
                          "scales": [10, 21, 45, 99, 153, 207, 261],
                          "aspect_ratio_0": [2,],
                          "aspect_ratio_1": [2, 3],
                          "aspect_ratio_2": [2, 3],
                          "aspect_ratio_3": [2, 3],
                          "aspect_ratio_4": [2,],
                          "aspect_ratio_5": [2,],
                          "nb_classes": 4,
        })
        test_dataset = get_test_dataset(args, dataset)
        test_dataloader = get_test_dataloader(test_dataset)

        ssd300 = SSD300(args)
        ssd300.cuda()

        if os.path.isfile(path_model):
            load_checkpoint(ssd300, path_model)
            checkpoint = torch.load(path_model,
                                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']

        ssd300.eval()

        dboxes = dboxes300_coco(args)
        encoder = Encoder(dboxes)

        N_gpu = 1
        ret = []

        inv_map = {v: k for k, v in test_dataset.label_map.items()}

        for nbatch, (img, img_id, img_size, _, _) in enumerate(test_dataloader):
            img_data = test_dataset.images[test_dataset.img_keys[nbatch]]
            fn = img_data[0]
            img_path = os.path.join(dataset, 'test/', fn)
            plt.imshow(Image.open(img_path).convert("RGB"))
            # tensor_image = tensor_image.view(tensor_image.shape[1], tensor_image.shape[2], tensor_image.shape[0])



            inp = img.cuda()
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

                htot, wtot = img_size[0][idx].item(), img_size[1][idx].item()
                loc, label, prob = [r.cpu().numpy() for r in result]
                count, count_wrong = 0, 0
                for loc_, label_, prob_ in zip(loc, label, prob):

                    if (inv_map[label_] == 0 and prob_ > 0.1) or (inv_map[label_] == 1 and prob_ > 0.6) or (inv_map[label_] == 2 and prob_ > 0.4):
                        count += 1
                        print(htot, wtot, inv_map[label_], (loc_[0] * wtot, loc_[1] * htot), (loc_[2] - loc_[0]) * wtot, (loc_[3] - loc_[1]) * htot)
                        color = ['r', 'g', 'b', 'y'][inv_map[label_]]
                        plt.gca().add_patch(Rectangle((loc_[0] * wtot, loc_[1] * htot), (loc_[2] - loc_[0]) * wtot, (loc_[3] - loc_[1]) * htot, linewidth=1, edgecolor=color, facecolor='none'))
                        # ret.append([img_id[idx], loc_[0] * wtot,
                        #             loc_[1] * htot,
                        #             (loc_[2] - loc_[0]) * wtot,
                        #             (loc_[3] - loc_[1]) * htot,
                        #             prob_,
                        #             inv_map[label_]])
                    else:
                        count_wrong += 1
                print(count, count_wrong)

            plt.savefig(f'{output_folder}/out_{nbatch}.jpg')
            plt.clf()





