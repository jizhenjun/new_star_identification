import pandas as pd
import cv2
is_new_star = {"newtarget": 1, "isstar": 1, "asteroid": 1, "isnova": 1, "known": 1, "noise": 0, "ghost": 0, "pity": 0,}
df = pd.read_csv("data/train.csv")
data = {'path':[], 'x1':[], 'y1':[], 'x2':[], 'y2':[], 'class_name':[]}
count = 0
for index, row in df.iterrows():
    fig = row['id']
    count += 1
    if (count%100 == 0):
        print("Loading image: ", count+1, ", ", fig)
        # break
    y = row['x']
    x = row['y']
    folder_name = fig[0:2]
    img_a = cv2.imread("data/af2019-cv-training-20190312/" + folder_name + "/" + fig + "_a.jpg")
    # img_b = cv2.imread("data/af2019-cv-training-20190312/" + folder_name + "/" + fig + "_b.jpg")
    # img_c = cv2.imread("data/af2019-cv-training-20190312/" + folder_name + "/" + fig + "_c.jpg")
    # img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    # img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    # img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    # img = cv2.merge([img_a, img_b, img_c])
    img = img_a
    height, width, channel = img.shape
    board = 4
    while height < board * 2 or width < board * 2:
        img = cv2.resize(img, (height * 2, width * 2))
        height, width, channel = img.shape
    x = max(board, x)
    x = min(height - board, x)
    y = max(board, y)
    y = min(width - board, y)

    save_path = "data/train_image/" + fig + ".png"
    # cv2.imwrite(save_path, img)
    data["path"].append(save_path)
    data["x1"].append(x - board)
    data["x2"].append(x + board)
    data["y1"].append(y - board)
    data["y2"].append(y + board)
    data["class_name"].append(is_new_star[row['judge']])
data = pd.DataFrame(data)
data.to_csv("data.csv", columns = ['path', 'x1', 'x2', 'y1', 'y2', 'class_name'], index=False,header=False)
    