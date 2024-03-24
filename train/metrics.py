import torch
import numpy as np

def closest_class(preds):
    class_resistancies = torch.tensor([0., 0.3, 0.15, 0.7, 0.02, 0.025]).reshape(1,6,1,1)
    preds_repeated = preds.unsqueeze(1).repeat(1, class_resistancies.shape[1], 1, 1)
    preds_diff = preds_repeated - class_resistancies
    preds_closest = torch.argmin(torch.abs(preds_diff), dim=1)
    return preds_closest