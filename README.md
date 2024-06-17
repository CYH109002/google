# ChatGPT fruit information identification
## 安裝Python
### 下載安裝包：

* 前往 [Python的官方網站](https://www.python.org/downloads/)。
* 下載適用於Windows的最新版本的Python安裝程序（通常是一個`.exe`文件）。
<img src="https://github.com/CYH109002/google/assets/129057021/97879023-725b-4a31-b032-98bddf54614d" alt="" style="width:80%;">

---

### 運行安裝程序：
* 雙擊下載的安裝程序。
* 勾選「Add Python to PATH」選項，這將允許你從命令行訪問Python。
* 點擊「Install Now」或「Customize Installation」（如果你需要自定義安裝）。
<img src="https://github.com/CYH109002/google/assets/129057021/914d50bf-47f0-4cdc-910b-181ad8734bc7" alt="" style="width:50%;">

---

### 驗證安裝：

* 打開命令提示符（Command Prompt）。
* 輸入`python --version`，如果安裝成功，你應該會看到已安裝的Python版本號。
<img src="https://github.com/CYH109002/google/assets/129057021/e561b8cb-6617-4d4b-8fc3-464890d59221" alt="" style="width:80%;">

---

## 安裝Tensorflow

### 用pip安裝Tensorflow
* 打開命令提示符（Command Prompt）。
* 輸入`pip install tensorflow==2.12.0`
<img src="https://github.com/CYH109002/google/assets/129057021/ebcea62c-0ac9-46bc-8212-0c40ede31e48" alt="" style="width:80%;">

---

### 檢查Tensorflow
* 輸入`pip show tensorflow`

<img src="https://github.com/CYH109002/google/assets/129057021/00a5d7ed-fed0-42ae-9e06-540057897990" alt="" style="width:80%;">

---

## 影像辨識模型訓練
### 使用Teachable Machine訓練AI
* 前往 [Teachable Machine的官方網站](https://teachablemachine.withgoogle.com/)。
* 選擇 **Image Project**
<img src="https://github.com/CYH109002/google/assets/129057021/3da9dc70-80be-4a1c-8926-e54c75139b21" alt="" style="width:80%;">

---

* 訓練完成後，點擊 **Export Model** 導出模型。
<img src="https://github.com/CYH109002/google/assets/129057021/83641e81-50cc-4be4-8b17-029348d68c77" alt="" style="width:80%;">

---

* 將完成訓練的模型下載

<img src="https://github.com/CYH109002/google/assets/129057021/7a8de4b8-e90e-45b9-b700-3ed303883b8a" alt="" style="width:50%;">

---

## 建立程式
### 安裝函式庫
* 打開命令提示符（Command Prompt）。
* 輸入`pip install opencv-python`
<img src="https://github.com/CYH109002/google/assets/129057021/c3764fdb-8566-4ab1-bb37-18f48c0a7923" alt="" style="width:80%;">

---

### 建立執行環境
* 建立一個資料夾
* 將剛下載好的檔案解壓縮，並放置資料夾中
* 在資料夾裡建立程式

<img src="https://github.com/CYH109002/google/assets/129057021/542c33c7-e929-4379-9cf8-7bb2ed65039a" alt="" style="width:80%;">

---

```python
# main.py 程式碼 
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

```

---

## 使用 ChatGPT API 
### 設置環境
* 打開命令提示符（Command Prompt）。
* 輸入`pip install openai`

<img src="https://github.com/CYH109002/google/assets/129057021/ba124a14-7489-4828-a168-a781dfc22c08" alt="" style="width:80%;">

---
* 依照[此影片](https://youtu.be/g4CAdpK6Q7Q?t=181)設置環境

<img src="https://github.com/CYH109002/google/assets/129057021/3cee5dbf-da1d-4f7e-a239-2ea3a069d7bf" alt="" style="width:65%;">

---

### 建立ChatGPT程式
* 建立一個名為 **ChatGPTAPI** 的python程式 

```python
# 
from openai import OpenAI
client = OpenAI()

def Connet_ChatGPT(fruit_name):
    print("")
    messages = []
    message = f"what is {fruit_name}"
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    desired_message = response.choices[0].message.content
    print(f"This is {fruit_name}")
    print(f"ChatGPT:{desired_message}")


```
