from modules.build_contextpath import build_contextpath

import sys, os

sys.path.append("/home/daniil/repos/pytorch-segmentation-detection/")
sys.path.append("/home/daniil/repos/pytorch-segmentation-detection/synchronized_batchnorm/")
sys.path.insert(0, '/home/daniil/repos/pytorch-segmentation-detection/vision/')

import torch.nn as nn
import torchvision.models as models
import torch

# from pytorch_segmentation_detection.datasets.pascal_voc import PascalVOCSegmentation

from pytorch_segmentation_detection.transforms import (ComposeJoint,
                                                       RandomHorizontalFlipJoint,
                                                       RandomScaleJoint,
                                                       CropOrPad,
                                                       ResizeAspectRatioPreserve)

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms

import numbers
import random

from matplotlib import pyplot as plt

import numpy as np
from PIL import Image

from sklearn.metrics import confusion_matrix

from dataset.voc import VOCSegmentationIncremental
import tasks
import argparser

def flatten_logits(logits, number_of_classes):
    """Flattens the logits batch except for the logits dimension"""

    logits_permuted = logits.permute(0, 2, 3, 1)
    logits_permuted_cont = logits_permuted.contiguous()
    logits_flatten = logits_permuted_cont.view(-1, number_of_classes)

    return logits_flatten


def flatten_annotations(annotations):
    return annotations.view(-1)


def get_valid_annotations_index(flatten_annotations, mask_out_value=255):
    return torch.squeeze(torch.nonzero((flatten_annotations != mask_out_value)), 1)


def adjust_learning_rate(optimizer, iteration):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    max_iteration = 10000.0

    multiplier = (1.0 - (iteration / max_iteration)) ** (0.9)

    lr = 0.0001 * multiplier

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


from pytorch_segmentation_detection.transforms import RandomCropJoint

class Resnet18_16s(nn.Module):

    def __init__(self, num_classes=1000):
        super(Resnet18_16s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet18_16s = models.resnet50(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True,
                                       additional_blocks=0)

        resnet18_16s.logits_conv_aux = nn.Conv2d(1024, num_classes, 1)
        resnet18_16s.logits_conv_final = nn.Conv2d(resnet18_16s.inplanes, num_classes, 1)

        self.resnet18_16s = resnet18_16s

    def forward(self, x):
        input_spatial_dim = x.size()[2:]

        x = self.resnet18_16s.conv1(x)
        x = self.resnet18_16s.bn1(x)
        x = self.resnet18_16s.relu(x)
        x = self.resnet18_16s.maxpool(x)

        x = self.resnet18_16s.layer1(x)
        x = self.resnet18_16s.layer2(x)
        x = self.resnet18_16s.layer3(x)

        aux_logits = self.resnet18_16s.logits_conv_aux(x)

        x = self.resnet18_16s.layer4(x)

        final_logits = self.resnet18_16s.logits_conv_final(x)

        aux_logits = nn.functional.upsample_bilinear(input=aux_logits, size=input_spatial_dim)
        final_logits = nn.functional.upsample_bilinear(input=final_logits, size=input_spatial_dim)

        return final_logits, aux_logits


from sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback


def make_batchnorm_syncronized(module):
    for child_module_name, child_module in module.named_children():

        if isinstance(child_module, nn.BatchNorm2d):
            sync_bn = SynchronizedBatchNorm2d(child_module.num_features)
            sync_bn.weight = child_module.weight
            sync_bn.bias = child_module.bias
            sync_bn.running_var = child_module.running_var
            sync_bn.running_mean = child_module.running_mean
            module.__setattr__(child_module_name, sync_bn)

def main(opts):
    # Define the validation function to track MIoU during the training
    def validate():
        fcn.eval()

        overall_confusion_matrix = None

        for image, annotation in valset_loader:

            image = Variable(image.cuda())
            logits, _ = fcn(image)

            # First we do argmax on gpu and then transfer it to cpu
            logits = logits.data
            _, prediction = logits.max(1)
            prediction = prediction.squeeze(1)

            prediction_np = prediction.cpu().numpy().flatten()
            annotation_np = annotation.numpy().flatten()

            # Mask-out value is ignored by default in the sklearn
            # read sources to see how that was handled

            current_confusion_matrix = confusion_matrix(y_true=annotation_np,
                                                        y_pred=prediction_np,
                                                        labels=labels)

            if overall_confusion_matrix is None:

                overall_confusion_matrix = current_confusion_matrix
            else:

                overall_confusion_matrix += current_confusion_matrix

        intersection = np.diag(overall_confusion_matrix)
        ground_truth_set = overall_confusion_matrix.sum(axis=1)
        predicted_set = overall_confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        intersection_over_union = intersection / union.astype(np.float32)
        mean_intersection_over_union = np.mean(intersection_over_union)

        fcn.train()

        return mean_intersection_over_union

    def validate_train():
        fcn.eval()

        overall_confusion_matrix = None

        for image, annotation in train_subset_loader:

            image = Variable(image.cuda())
            logits, _ = fcn(image)

            # First we do argmax on gpu and then transfer it to cpu
            logits = logits.data
            _, prediction = logits.max(1)
            prediction = prediction.squeeze(1)

            prediction_np = prediction.cpu().numpy().flatten()
            annotation_np = annotation.numpy().flatten()

            # Mask-out value is ignored by default in the sklearn
            # read sources to see how that was handled

            current_confusion_matrix = confusion_matrix(y_true=annotation_np,
                                                        y_pred=prediction_np,
                                                        labels=labels)

            if overall_confusion_matrix is None:

                overall_confusion_matrix = current_confusion_matrix
            else:

                overall_confusion_matrix += current_confusion_matrix

        intersection = np.diag(overall_confusion_matrix)
        ground_truth_set = overall_confusion_matrix.sum(axis=1)
        predicted_set = overall_confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        intersection_over_union = intersection / union.astype(np.float32)
        mean_intersection_over_union = np.mean(intersection_over_union)

        fcn.train()

        return mean_intersection_over_union

    train_transform = ComposeJoint(
        [
            RandomHorizontalFlipJoint(),
            RandomCropJoint(crop_size=(513, 513)),
            # [ResizeAspectRatioPreserve(greater_side_size=384),
            # ResizeAspectRatioPreserve(greater_side_size=384, interpolation=Image.NEAREST)],

            # RandomCropJoint(size=(274, 274))
            # RandomScaleJoint(low=0.9, high=1.1),

            # [CropOrPad(output_size=(288, 288)), CropOrPad(output_size=(288, 288), fill=255)],
            [transforms.ToTensor(), None],
            [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), None],
            [None, transforms.Lambda(lambda x: torch.from_numpy(np.asarray(x)).long())]
        ])

    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)

    train_dst = VOCSegmentationIncremental(root=opts.data_root, train=True, transform=train_transform,
                                           labels=list(labels), labels_old=list(labels_old),
                                           idxs_path=path_base + f"/train-{opts.step}.npy",
                                           masking=not opts.no_mask, overlap=opts.overlap)

    valid_transform = ComposeJoint(
        [
            [transforms.ToTensor(), None],
            [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), None],
            [None, transforms.Lambda(lambda x: torch.from_numpy(np.asarray(x)).long())]
        ])

    val_dst = VOCSegmentationIncremental(root=opts.data_root, train=False, transform=valid_transform,
                                        labels=list(labels), labels_old=list(labels_old),
                                        idxs_path=path_base + f"/val-{opts.step}.npy",
                                        masking=not opts.no_mask, overlap=True)

    trainloader = torch.utils.data.DataLoader(train_dst, batch_size=opts.batch_size,
                                   shuffle=True, num_workers=opts.num_workers, drop_last=True)
    valset_loader = torch.utils.data.DataLoader(val_dst, batch_size=opts.batch_size if opts.crop_val else 1,
                                    shuffle=True, num_workers=opts.num_workers)

    train_subset_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(904))
    train_subset_loader = torch.utils.data.DataLoader(dataset=train_dst, batch_size=1,
                                                      sampler=train_subset_sampler,
                                                      num_workers=2)

    fcn = Resnet18_16s(num_classes=21)
    fcn.apply(make_batchnorm_syncronized)
    fcn = DataParallelWithCallback(fcn, device_ids=[0, 1, 2, 3], output_device=3)

    fcn.cuda()
    fcn.train()

    final_criterion = nn.CrossEntropyLoss(size_average=False).cuda(3)
    aux_criterion = nn.CrossEntropyLoss(size_average=False).cuda(3)

    optimizer = optim.Adam(fcn.parameters(), lr=0.0001)

    best_validation_score = 0
    loss_current_iteration = 0

    loss_history = []
    loss_iteration_number_history = []

    validation_current_iteration = 0
    validation_history = []
    validation_iteration_number_history = []

    train_validation_current_iteration = 0
    train_validation_history = []
    train_validation_iteration_number_history = []

    iter_size = 20

    for epoch in range(1000):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):

            # get the inputs
            img, anno = data

            # We need to flatten annotations and logits to apply index of valid
            # annotations. All of this is because pytorch doesn't have tf.gather_nd()
            anno_flatten = flatten_annotations(anno)
            index = get_valid_annotations_index(anno_flatten, mask_out_value=255)
            anno_flatten_valid = torch.index_select(anno_flatten, 0, index)

            # wrap them in Variable
            # the index can be acquired on the gpu
            img, anno_flatten_valid, index = Variable(img.cuda()), Variable(anno_flatten_valid.cuda(3)), Variable(
                index.cuda(3))

            # zero the parameter gradients
            optimizer.zero_grad()

            adjust_learning_rate(optimizer, loss_current_iteration)

            # forward + backward + optimize
            final_logits, aux_logits = fcn(img)

            final_logits_flatten = flatten_logits(final_logits, number_of_classes=21)
            final_logits_flatten_valid = torch.index_select(final_logits_flatten, 0, index)
            loss = final_criterion(final_logits_flatten_valid, anno_flatten_valid)

            aux_logits_flatten = flatten_logits(aux_logits, number_of_classes=21)
            aux_logits_flatten_valid = torch.index_select(aux_logits_flatten, 0, index)
            loss += 0.4 * aux_criterion(aux_logits_flatten_valid, anno_flatten_valid)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += (loss.data[0] / (final_logits_flatten.size(0) * 2))
            if i % 2 == 1:
                loss_history.append(running_loss / 2)
                loss_iteration_number_history.append(loss_current_iteration)

                loss_current_iteration += 1
                """
                loss_axis.lines[0].set_xdata(loss_iteration_number_history)
                loss_axis.lines[0].set_ydata(loss_history)

                loss_axis.relim()
                loss_axis.autoscale_view()
                loss_axis.figure.canvas.draw()
                """
                loss_current_iteration += 1

                running_loss = 0.0

        current_validation_score = validate()
        validation_history.append(current_validation_score)
        validation_iteration_number_history.append(validation_current_iteration)

        validation_current_iteration += 1

        #validation_axis.lines[0].set_xdata(validation_iteration_number_history)
        #validation_axis.lines[0].set_ydata(validation_history)

        current_train_validation_score = validate_train()
        train_validation_history.append(current_train_validation_score)
        train_validation_iteration_number_history.append(train_validation_current_iteration)

        train_validation_current_iteration += 1

        #validation_axis.lines[1].set_xdata(train_validation_iteration_number_history)
        #validation_axis.lines[1].set_ydata(train_validation_history)

        #validation_axis.relim()
        #validation_axis.autoscale_view()
        #validation_axis.figure.canvas.draw()

        # Save the model if it has a better MIoU score.
        if current_validation_score > best_validation_score:
            torch.save(fcn.state_dict(), 'resnet_101_psp_check.pth')
            best_validation_score = current_validation_score
            print(best_validation_score)


    print(f"Epoch {epoch} | current_validation_score : {current_validation_score} | ")

    print('Finished Training')


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    os.makedirs("checkpoints/step", exist_ok=True)

    main(opts)

