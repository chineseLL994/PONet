



import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from net import Mnet
# from HWSI_final_version.HWSI_final_code.net import Mnet

from lib.dataset import Data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------------
# 你的 CAM 可视化（原样集成）
# ---------------------------
# def CAM(features, img_path, save_path):
#     features.retain_grad()
#     _ = features.grad  # 该实现不使用 grads，仅确保可访问
#     features = features.squeeze(0)  # [C,H,W]
#
#     heatmap = features.detach().cpu().numpy()
#     heatmap = np.mean(heatmap, axis=0)
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= (np.max(heatmap) + 1e-8)
#
#     img = cv2.imread(img_path)  # BGR
#     if img is None:
#         raise FileNotFoundError(f"Cannot read image: {img_path}")
#
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     superimposed_img = np.uint8(heatmap * 0.5 + img * 0.5)
#     cv2.imwrite(save_path, superimposed_img)

# ---------------------------
# 你的 CAM 可视化（加入透明度参数）
# ---------------------------
def CAM(features, img_path, save_path, alpha: float = 0.35, save_heat: bool = False, heat_path: str = None,
        colormap: int = cv2.COLORMAP_JET):
    """
    将 features 生成的 CAM 叠加到原图上并保存
    alpha: 叠加透明度（0~1），越大热力图越明显
    save_heat: 是否另存一张纯热力图
    heat_path: 纯热力图保存路径（当 save_heat=True 时必填）
    colormap: 伪彩色映射（默认 JET，可改为 cv2.COLORMAP_TURBO 等）
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # 1) 生成 [H,W] 的归一化热力图
    features.retain_grad()
    _ = features.grad  # 不使用 grads，这行只是确保 OK
    cam = features.squeeze(0).detach().cpu().numpy()  # [C,H,W] -> [C,H,W]
    cam = cam.mean(axis=0)                            # 通道平均 -> [H,W]
    cam = np.maximum(cam, 0)
    cam /= (cam.max() + 1e-8)

    # 2) 读取原图并对齐尺寸
    img = cv2.imread(img_path)  # BGR
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    H, W = img.shape[:2]
    cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)

    # 3) 伪彩色映射
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)

    # (可选) 单独保存热力图
    if save_heat:
        if heat_path is None:
            raise ValueError("save_heat=True 时必须提供 heat_path")
        cv2.imwrite(heat_path, heatmap)

    # 4) 透明度叠加（推荐方式）
    overlay = cv2.addWeighted(src1=img, alpha=1.0 - alpha, src2=heatmap, beta=alpha, gamma=0)

    cv2.imwrite(save_path, overlay)


# ---------------------------
# 判定“非简化卷积层”：Conv2d 且 kernel!=1x1 且 groups==1
# ---------------------------
def is_non_simplified_conv(m: nn.Module) -> bool:
    if not isinstance(m, nn.Conv2d):
        return False
    k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
    return (k != (1, 1)) and (m.groups == 1)

# ---------------------------
# 动态追踪：找“实际执行的最后一个非简化卷积层”
# 用一次无梯度前向记录执行顺序，避免选到未走通路的层
# ---------------------------
def trace_last_used_non_simplified_conv(model: nn.Module, rgb, t, d) -> nn.Conv2d:
    fired = []
    handles = []

    def tracer(mod, inp, out):
        fired.append(mod)

    # 给所有 Conv2d 挂临时 hook
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(tracer))

    # 一次“探路前向”（不建图即可）
    with torch.no_grad():
        _ = model(rgb, t, d)

    # 移除临时 hook
    for h in handles:
        h.remove()

    # 从后往前找“非简化卷积层”
    for m in reversed(fired):
        if is_non_simplified_conv(m):
            return m

    # 兜底：若未找到，至少返回最后一个被执行的 Conv2d
    if fired:
        return fired[-1]
    raise RuntimeError("未在前向路径中找到任何 Conv2d，无法用于 CAM。")

# ---------------------------
# 主程序
# ---------------------------
if __name__ == '__main__':
    model_path = 'D:\Chinese\HWSI\Revise\checkpoints\epoch_400.pth'
    data_root  = 'D:/Chinese/HWSI/VDT-dataset/Test/'
    # out_path   = 'PONet_Result/SOD_Evaluation_Metrics/PreOut/revise/HFMD/'
    cam_path   = 'D:/Chinese/HWSI/PONet_Result/Attention_maps/PONet_rgbdtg/'
    os.makedirs(cam_path, exist_ok=True)

    dataset = Data(root=data_root, mode='test')
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    net = Mnet().cuda().eval()
    print(f'loading model from {model_path} ...')
    net.load_state_dict(torch.load(model_path))

    # 保存特征的容器（避免 nonlocal/global 报错）
    feat_holder = {"feats": None}
    target_layer = None
    fwd_handle = None

    # 原图目录（按你的 Data 类实际字段改）
    rgb_dir = getattr(dataset, 'rgb_root', os.path.join(data_root, 'V'))

    time_s = time.time()
    img_num = len(loader)

    for i, (rgb, t, d, _, (H, W), name) in enumerate(loader):
        rgb = rgb.cuda().float()
        t   = t.cuda().float()
        d   = d.cuda().float()

        # 第一次迭代：动态追踪找到“最后一个非简化且实际执行的 Conv2d”
        if target_layer is None:
            target_layer = trace_last_used_non_simplified_conv(net, rgb, t, d)
            print("Using target layer for CAM:", target_layer)

            # 注册正式的 forward hook
            def fwd_hook(module, inp, out):
                feat_holder["feats"] = out
                feat_holder["feats"].retain_grad()
            fwd_handle = target_layer.register_forward_hook(fwd_hook)

        # 正式前向（需要梯度）
        score, score2, score3, score4, s1_sig, s2_sig, s3_sig, s4_sig, rgbdt_g, rgbdt4, rgbdt3, rgbdt2, rgbdt1 = net(rgb, t, d)
        # score, score2, score3, score4, score5, s_sig, s2_sig, s3_sig, s4_sig, s5_sig = net(rgb, t, d)

        # 反向目标（无类别）：显著图平均
        net.zero_grad(set_to_none=True)
        target_scalar = torch.sigmoid(rgbdt_g).mean()
        target_scalar.backward(retain_graph=True)

        features = feat_holder["feats"]
        if features is None:
            raise RuntimeError("Forward hook 未捕获特征，请检查 target_layer 是否在前向通路。")

        # 保存
        base = name[0][3:-4]
        cam_save_path = os.path.join(cam_path, f'{base}_cam.png')
        img_path = os.path.join(rgb_dir, name[0][3:])

        CAM(features, img_path, cam_save_path)
        print(f"[{base}] CAM overlay saved -> {cam_save_path}")

    # 清理句柄
    if fwd_handle is not None:
        fwd_handle.remove()

    time_e = time.time()
    print('speed: %f FPS' % (img_num / (time_e - time_s)))




