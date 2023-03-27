import torch
import torch.nn as nn

def load_model_dict(model, pretrained_dict):
    """ load pretrained state dict, keys may not match with model

    Args:
        model: nn.Module

        pretrained_dict: collections.OrderedDict
    
    """
    # 1. filter out unnecessary keys
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=0.1)
        if hasattr(m.bias, 'data'):
            nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=0.1)
        # if hasattr(m, 'bias'):
        #     nn.init.constant_(m.bias, 0)

    # elif isinstance(m, nn.BatchNorm2d):
    #     nn.init.xavier_normal_(m.weight, gain=0.05)
    #     nn.init.constant_(m.bias, 0)

def rename_model_dict_keys(pretrained_dict_path, rename_dict):
    """ load pretrained state dict, keys may not match with model

    Args:
        model: nn.Module

        pretrained_dict: collections.OrderedDict
    
    """
    pretrained_dict = torch.load(pretrained_dict_path)
    # 1. filter out unnecessary keys
    for oldname, newname in rename_dict.items():
        pretrained_dict[newname] = pretrained_dict[oldname]
        pretrained_dict.pop(oldname)
    torch.save(pretrained_dict, pretrained_dict_path)


if __name__ == "__main__":
    dict_path = "/GPFS/rhome/yifanlu/OpenCOOD/opencood/logs/OPV2V_lift_splat_shoot_gt_depth_8x_train_single_infer_max_fusion/net_epoch_bestval_at55.pth"
    rename_dict = {"camencode.net.weight": "camencode.image_head.weight",
                   "camencode.net.bias": "camencode.image_head.bias"}
    rename_model_dict_keys(dict_path, rename_dict)