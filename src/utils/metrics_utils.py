import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def centroid_to_bbox(centroids, w = 50/256, h = 50/256):
    x0_y0 = centroids - torch.tensor([w/2, h/2]).to(device)
    x1_y1 = centroids + torch.tensor([w/2, h/2]).to(device)
    return torch.cat([x0_y0, x1_y1], dim = 1)

# Function to compute the IoU for a batch of labels
from torchvision.ops import box_iou
def iou_batch(output_labels, target_labels):
    output_bbox = centroid_to_bbox(output_labels)
    target_bbox = centroid_to_bbox(target_labels)
    return torch.trace(box_iou(output_bbox, target_bbox)).item()