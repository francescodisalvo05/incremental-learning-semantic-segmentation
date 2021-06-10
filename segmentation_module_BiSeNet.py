import torch
import torch.nn as nn

import torch.nn.functional as functional

from functools import reduce

from modules.build_BiSeNet import  BiSeNet

def make_model(opts, classes=None):

    # string with the backbone e.g. 'resnet'
    body = opts.backbone

    if not opts.no_pretrained:
        pretrained_path = f'pretrained/{opts.backbone}_{opts.norm_act}.pth.tar'
        pre_dict = torch.load(pretrained_path, map_location='cpu')
        del pre_dict['state_dict']['classifier.fc.weight']
        del pre_dict['state_dict']['classifier.fc.bias']

        body.load_state_dict(pre_dict['state_dict'])
        del pre_dict  # free memory

    head = BiSeNet(body)

    if classes is not None:
        model = IncrementalSegmentationBiSeNet(body, head, classes=classes, fusion_mode=opts.fusion_mode)
    else:
        # model = SegmentationModule(body, head, head_channels, opts.num_classes, opts.fusion_mode)
        pass

    return model


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalSegmentationBiSeNet(nn.Module):

    def __init__(self, body, head, classes, ncm=False, fusion_mode="mean"):
        super(IncrementalSegmentationBiSeNet, self).__init__()

        self.body = body
        self.head = head

        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"

        # classifiers supervision 1
        self.supervision1 = nn.ModuleList(
            [nn.Conv2d(in_channels=1024, out_channels=c, kernel_size=1) for c in classes]
        )
        # classifiers supervision 2
        self.supervision2 = nn.ModuleList(
            [nn.Conv2d(in_channels=2048, out_channels=c, kernel_size=1) for c in classes]
        )

        # classifiers for the final layers
        self.cls = nn.ModuleList(
            [nn.Conv2d(in_channels=256, out_channels=c, kernel_size=1) for c in classes]
            # [nn.Conv2d(256, c, 1) for c in classes]
        )

        self.classes = classes
        self.head_channels = 256
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.means = None

    def _network(self, x, ret_intermediate=False):

        result, cx1, cx2 = self.head(x)
        out = []
        cx1_out = []
        cx2_out = []

        for mod in self.cls:
            out.append(mod(result))

        for mod in self.supervision1:
            cx1_out.append(mod(cx1))

        for mod in self.supervision2:
            cx2_out.append(mod(cx2))

        x_o = torch.cat(out, dim=1)
        cx1_sup = torch.cat(cx1_out, dim=1)
        cx2_sup = torch.cat(cx2_out, dim=1)

        # it is forced to True at the moment
        if ret_intermediate:
            return x_o, cx1_sup, cx2_sup

        return x_o

    def init_new_classifier(self, device):
        cls = self.cls[-1]

        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x, scales=None, do_flip=False, ret_intermediate=False):
        out_size = x.shape[-2:]

        if ret_intermediate:
            out, out_cx1, out_cx2 = self._network(x, ret_intermediate)
            out = functional.interpolate(out, size=out_size, mode="bilinear", align_corners=False)
            out_1 = functional.interpolate(out_cx1, size=out_size, mode='bilinear', align_corners=False)
            out_2 = functional.interpolate(out_cx2, size=out_size, mode='bilinear', align_corners=False)
            return out, out_1, out_2

        return functional.interpolate(self._network(x, ret_intermediate), size=out_size, mode="bilinear", align_corners=False)

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
