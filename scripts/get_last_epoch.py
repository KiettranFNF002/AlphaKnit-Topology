import os
import re

ckpt_dir = "checkpoints"

def get_last_epoch():
    if not os.path.exists(ckpt_dir):
        return -1
        
    epochs = []
    # Match both epoch_011.pt and checkpoint_xyz_epoch_011.pt
    pattern = re.compile(r".*epoch_(\d+)\.pt")
    
    for f in os.listdir(ckpt_dir):
        m = pattern.match(f)
        if m:
            epochs.append(int(m.group(1)))
            
    if not epochs:
        return -1
    return max(epochs)

if __name__ == "__main__":
    print(get_last_epoch())
