from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def show_image(img, label):
    w, h = img.size
    ImageDraw.Draw(img).rectangle([(label[0]-h*0.1, label[1]-h*0.1), (label[0]+h*0.1, label[1]+h*0.1)])
    plt.imshow(img)
    plt.show()

def show_image_with_2_bbox(image, label, target_label, bbox=(50, 50), thickness=2):
    w, h = bbox
    c_x , c_y = label
    c_x_target , c_y_target = target_label
    image = image.copy()
    ImageDraw.Draw(image).rectangle(((c_x-w//2, c_y-h//2), (c_x+w//2, c_y+h//2)), outline='green', width=thickness)
    ImageDraw.Draw(image).rectangle(((c_x_target-w//2, c_y_target-h//2), (c_x_target+w//2, c_y_target+h//2)), outline='red', width=thickness)
    plt.imshow(image)