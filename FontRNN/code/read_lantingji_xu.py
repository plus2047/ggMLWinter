#%%
import numpy as np
import re
from pathlib import Path
from PIL import Image


def read_lantingji_xu(image_dir="Lantingji_Xu"):
    images = [Image.open(f).convert("L") for f in sorted(Path(image_dir).glob("*.jpg"))]
    mat_list = [np.asarray(i) for i in images]

    text = open(image_dir + "/text.txt").read()
    text = re.sub(r"(\s+)|(\(.*?\))", "", text)

    labels = []
    valid = []
    idx = 0

    while idx < len(text):
        t = text[idx]
        if t != '[':
            labels.append(t)
            valid.append(True)
            idx += 1
        else:
            labels.append(text[idx + 1])
            valid.append(False)
            idx += 3
    
    return mat_list, "".join(labels), valid


#%%
if __name__ == "__main__":
    m, t, v = read_lantingji_xu()
    print(t)
