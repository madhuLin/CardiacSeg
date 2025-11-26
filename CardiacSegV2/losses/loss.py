import torch
from monai.losses import DiceCELoss, DiceFocalLoss, BoundaryLoss
from losses.cotr_loss import CoTrLoss

# ---------------------------------------------------
# 自訂：根據 config 產生 class weights
# ---------------------------------------------------
def get_class_weights(num_classes, myocardium_idx=2, myo_weight=2.0):
    """
    num_classes: 幾類（通常 4: background, LV, MYO, RV）
    myocardium_idx: 心肌的 class index
    myo_weight: 給心肌的額外權重
    """
    w = torch.ones(num_classes)

    # 背景權重降低
    w[0] = 0.2

    # 心肌加權
    w[myocardium_idx] = myo_weight

    return w


# ---------------------------------------------------
# 主 loss function
# ---------------------------------------------------
def loss_func(loss_name, device="cuda", myo_weight=2.0):
    print(f'loss: {loss_name}, myocardium weight = {myo_weight}')

    class_weights = get_class_weights(
        num_classes=4,
        myocardium_idx=2,   # 心肌 class index
        myo_weight=myo_weight
    ).to(device)

    # ---------------------------------------------------
    # 基本的 Dice + CE
    # ---------------------------------------------------
    base_loss = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        weight=class_weights
    )

    # ---------------------------------------------------
    # 選擇模型 loss
    # ---------------------------------------------------
    if loss_name == "dice_ce":
        return base_loss

    elif loss_name == "dice_ce_focal":
        # DiceCE + Focal（可以讓邊界和小區域更準）
        return base_loss + 0.5 * DiceFocalLoss(
            to_onehot_y=True,
            softmax=True
        )

    elif loss_name == "dice_ce_boundary":
        # DiceCE + BoundaryLoss（心肌邊界更清晰）
        return base_loss + 0.1 * BoundaryLoss()

    elif loss_name == "cotr":
        # CoTr 專用
        return CoTrLoss(base_loss)

    else:
        raise ValueError(f'not found loss name: {loss_name}')
