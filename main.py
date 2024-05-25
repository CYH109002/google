import time,keyboard

#使用tensorflow
from keras.models import load_model  # 必须安装 TensorFlow 才能使用 Keras
import cv2  # 安装 opencv-python
import numpy as np

#使用ChatGPT
from openai import OpenAI
client = OpenAI()
from ChatGPTAPI import Connet_ChatGPT#匯入我寫的函式"Connet_ChatGPT"


np.set_printoptions(suppress=True)  # 禁用科学计数法以提高可读性
model = load_model("keras_Model.h5", compile=False)  # 加载模型
class_names = open("labels.txt", "r").readlines()  # 加载标签
camera = cv2.VideoCapture(0)  # 摄像头可能是 0 或 1，根据你计算机的默认摄像头设置


time.sleep(5)
if __name__ == '__main__':
    while True:
        
        ret, image = camera.read()  # 抓取摄像头图像

        #處理鏡頭畫面
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imshow("Webcam Image", image)  # 在窗口中显示图像
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1
        
        #偵測
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        # 輸出結果
        print("Fruit:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        fruit = class_name[2:]
    
        keyboard_input = cv2.waitKey(1)#讀取鍵盤
        if keyboard_input == 13:#13是 "Enter鍵"
            Connet_ChatGPT(class_name[2:])
            time.sleep(3) 
        if keyboard_input == 27:#27是 "Esc鍵"
            break
camera.release()
cv2.destroyAllWindows()

