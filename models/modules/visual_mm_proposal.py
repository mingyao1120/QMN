import torch


def calculate_iou(bbox1, bbox2):
    x1, w1 = bbox1[:, 0], bbox1[:, 1]
    x2, w2 = bbox2[:, 0], bbox2[:, 1]


    left1, right1 = torch.clamp(x1 - 0.5 * w1, min=0), torch.clamp(x1 + 0.5 * w1, max=1)
    left2, right2 = torch.clamp(x2 - 0.5 * w2, min=0), torch.clamp(x2 + 0.5 * w2, max=1)


    intersection_left = torch.max(left1, left2)
    intersection_right = torch.min(right1, right2)


    intersection_width = torch.clamp(intersection_right - intersection_left, min=0)


    union_width = w1 + w2 - intersection_width
    iou = intersection_width / union_width
    return iou # ,left1, right1, left2, right2



def match_and_fuse_params(gauss_param, my_param, ratio):
    num_samples, num_bboxes, _ = gauss_param.shape


    matched_params = torch.zeros_like(gauss_param)

    for i in range(num_samples): # 取batch

        iou_matrix  = calculate_iou(gauss_param[i], my_param[i])

        match_indices = torch.argmax(iou_matrix, dim=0)


        matched_params[i] = my_param[i].gather(dim=0, index=match_indices.view(1, -1).expand(num_bboxes, -1))


    fused_params = gauss_param*ratio + matched_params*(1 - ratio)

    return fused_params


