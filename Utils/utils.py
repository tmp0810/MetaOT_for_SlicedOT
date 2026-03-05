import os
import matplotlib.pyplot as plt
import numpy as np

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def insert(self, key, value):
        self[key] = value

def img_scale(im, im_shape):
    im = im if isinstance(im, np.ndarray) else im.data.cpu().numpy()
    im = im - im.min()
    im_max = im.max()
    # Chống chia cho 0 nếu ảnh là 1 màu đồng nhất
    if im_max > 0:
        im = im / im_max
    im = np.reshape(im, im_shape)
    return im

def save_r(imgs, x_a, x_b, path, title):
    # Đặt kích thước figure dài ra một chút để chứa 13 ảnh cho khỏi bị ép nén
    fig = plt.figure(figsize=(2 * (len(imgs) + 2), 2))
    fig.suptitle(title)
    
    # Vẽ chuỗi ảnh nội suy ở giữa
    for i, im in enumerate(imgs):
        ax = fig.add_subplot(1, len(imgs) + 2, i + 2) # i + 2 vì ảnh source chiếm vị trí 1
        ax.imshow(im, cmap='gray', vmin=0, vmax=1)
        ax.axis('off') # Tắt trục tọa độ cho ảnh đẹp hơn
        
    # Vẽ ảnh Nguồn (Source) ở vị trí đầu tiên
    ax = fig.add_subplot(1, len(imgs) + 2, 1)
    ax.imshow(img_scale(x_a, imgs[0].shape), cmap='gray', vmin=0, vmax=1)
    ax.set_title("Source")
    ax.axis('off')
    
    # Vẽ ảnh Đích (Target) ở vị trí cuối cùng
    ax = fig.add_subplot(1, len(imgs) + 2, len(imgs) + 2)
    ax.imshow(img_scale(x_b, imgs[0].shape), cmap='gray', vmin=0, vmax=1)
    ax.set_title("Target")
    ax.axis('off')
    
    plt.tight_layout() # Tự động căn chỉnh lề chống đè chữ
    
    # SỬA LỖI Ở ĐÂY: Lưu ảnh trước khi show!
    os.makedirs(path, exist_ok=True) # Đảm bảo thư mục tồn tại
    plt.savefig(os.path.join(path, "OT_%s.png" % (title)), bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()

def save_r_cons(x_a, x_b, y_a, y_b, path, title):
    fig = plt.figure(figsize=(12, 3))
    fig.suptitle(title)
    
    ax = fig.add_subplot(1, 4, 1)
    ax.axis('off')
    ax.imshow(x_a, cmap='gray') 
    
    ax = fig.add_subplot(1, 4, 2)
    ax.axis('off')
    ax.imshow(x_b, cmap='gray') 
    
    ax = fig.add_subplot(1, 4, 3)
    ax.axis('off')
    ax.imshow(y_a, cmap='gray') 
    
    ax = fig.add_subplot(1, 4, 4)
    ax.axis('off')
    ax.imshow(y_b, cmap='gray') 
    
    # SỬA LỖI Ở ĐÂY
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "OT_%s.png" % (title)), bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()
