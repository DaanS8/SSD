# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import time
import numpy as np
from contextlib import redirect_stdout
import io
from torch.autograd import Variable

from pycocotools.cocoeval import COCOeval


def evaluate(epoch, model, loss_func, coco, cocoGt, encoder, inv_map, args):
    loss_acc = 0
    counter = 0

    if args.distributed:
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    model.eval()
    if not args.no_cuda:
        model.cuda()
    ret = []
    classes = ['ball', 'hand', 'figure']
    start = time.time()

    with torch.no_grad():
    # for idx, image_id in enumerate(coco.img_keys):
        for nbatch, (img, img_id, img_size, gboxes, glabels) in enumerate(coco):
            if loss_func is not None:
                gbox_shape = (img.shape[0], encoder.dboxes.shape[0], encoder.dboxes.shape[1])
                glabel_shape = (img.shape[0], encoder.dboxes.shape[0])

                bbox, label = torch.empty(gbox_shape), torch.empty(glabel_shape)

                if not args.no_cuda:
                    if 'cuda' not in str(encoder.dboxes.device):
                        encoder.dboxes = encoder.dboxes.cuda()
                    img = img.cuda()
                    gboxes = gboxes.cuda()
                    glabels = glabels.cuda()
                    bbox = bbox.cuda()
                    label.cuda()

                for i, (gbox, glabel) in enumerate(zip(gboxes, glabels)):
                    first_index_background = (glabel == 0).nonzero(as_tuple=True)[0][0]
                    if first_index_background >= 200 or first_index_background < 0:
                        print(f'{first_index_background} = index of first background class, should be between 0 and 199')

                    gbox = gbox[:first_index_background]
                    glabel = glabel[:first_index_background]

                    enc_bbox, enc_label = encoder.encode(gbox, glabel)
                    bbox[i] = enc_bbox
                    label[i] = enc_label
                label = label.type(torch.cuda.LongTensor)

            elif not args.no_cuda:
                img = img.cuda()

            counter += 1
            print("Parsing batch: {}/{}".format(nbatch, len(coco)), end='\r')

            with torch.cuda.amp.autocast(enabled=args.amp):
                # Get predictions
                ploc, plabel = model(img)
            ploc, plabel = ploc.float(), plabel.float()

            if loss_func is not None:
                trans_bbox = bbox.transpose(1, 2).contiguous().cuda()
                gloc = Variable(trans_bbox, requires_grad=False)
                glabel = Variable(label, requires_grad=False)

                loss = loss_func(ploc, plabel, gloc, glabel)
                loss_acc += float(loss)

            if epoch in args.evaluation:
                # Handle the batch of predictions produced
                # This is slow, but consistent with old implementation.
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
                    for loc_, label_, prob_ in zip(loc, label, prob):
                        ret.append([img_id[idx], loc_[0] * wtot,
                                    loc_[1] * htot,
                                    (loc_[2] - loc_[0]) * wtot,
                                    (loc_[3] - loc_[1]) * htot,
                                    prob_,
                                    inv_map[label_]])
        E = None
        if epoch in args.evaluation:
            # Now we have all predictions from this rank, gather them all together
            # if necessary
            ret = np.array(ret).astype(np.float32)
            # Multi-GPU eval
            if args.distributed:  # DISBALE!!!
                raise Exception('Dont use, or reajust based on original repo')
                # NCCL backend means we can only operate on GPU tensors
                ret_copy = torch.tensor(ret).cuda()
                # Everyone exchanges the size of their results
                ret_sizes = [torch.tensor(0).cuda() for _ in range(N_gpu)]

                torch.cuda.synchronize()
                torch.distributed.all_gather(ret_sizes, torch.tensor(ret_copy.shape[0]).cuda())
                torch.cuda.synchronize()

                # Get the maximum results size, as all tensors must be the same shape for
                # the all_gather call we need to make
                max_size = 0
                sizes = []
                for s in ret_sizes:
                    max_size = max(max_size, s.item())
                    sizes.append(s.item())

                # Need to pad my output to max_size in order to use in all_gather
                ret_pad = torch.cat([ret_copy, torch.zeros(max_size - ret_copy.shape[0], 7, dtype=torch.float32).cuda()])

                # allocate storage for results from all other processes
                other_ret = [torch.zeros(max_size, 7, dtype=torch.float32).cuda() for i in range(N_gpu)]
                # Everyone exchanges (padded) results

                torch.cuda.synchronize()
                torch.distributed.all_gather(other_ret, ret_pad)
                torch.cuda.synchronize()

                # Now need to reconstruct the _actual_ results from the padded set using slices.
                cat_tensors = []
                for i in range(N_gpu):
                    cat_tensors.append(other_ret[i][:sizes[i]][:])

                final_results = torch.cat(cat_tensors).cpu().numpy()
            else:
                # Otherwise full results are just our results
                final_results = ret

            if args.local_rank == 0:
                print("")
                print("Predicting Ended, total time: {:.2f} s".format(time.time() - start))

            print('Global Stats')
            cocoDt = cocoGt.loadRes(final_results)
            E = COCOeval(cocoGt, cocoDt, iouType='bbox')
            E.params.catIds = [0, 1, 2]
            E.evaluate()
            E.accumulate()

            if args.local_rank == 0:
                E.summarize()
                print("Current AP: {:.5f}".format(E.stats[0]))

                # Based on https://github.com/cocodataset/cocoapi/issues/381

                all_precision = E.eval['precision']
                print(np.mean(all_precision[0, :, :, 0, 2], axis=1).tolist())
                print(all_precision[0, :, 0, 0, 2].tolist())
                print(all_precision[0, :, 1, 0, 2].tolist())
                print(all_precision[0, :, 2, 0, 2].tolist())



            else:
                # fix for cocoeval indiscriminate prints
                with redirect_stdout(io.StringIO()):
                    E.summarize()






    model.train()
    loss_av = loss_acc/counter
    print(f'Evaluation loss: {loss_av}')
    return E.stats[0] if E is not None else -1, loss_av  # Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]

