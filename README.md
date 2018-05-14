# HCI_lite
> A lite human computer interaction system, mainly dependent on you hand in your webcam.
> Author: Peng Zheng.

## Dependencies:

    1. OpenCV==3.4.0
    2. numpy==1.14.3
    3. tensorflow-gpu==1.8.0    # CUDA=9, CUDNN=7.


## config:
```python3
video mode setting: {
"display": Default value, of which ink would fade, with tracking effect,
"writing": Ink would not fade,
"styleTransfer": Stylize the whole input from webcam or only your clothes.
"calc": Do math evaluation from handwritten formula by OCR tech.
      "evaluation": evaluate the result of handwritten formula.
}
```
# Mode:

- ### Guide:

```python3
video mode setting: {
"display": Default value, of which ink would fade, with tracking effect,
"writing": Ink would not fade,
"styleTransfer": Stylize the whole input from webcam or only your clothes.
"calc": Do math evaluation from handwritten formula by OCR tech.
      "evaluation": evaluate the result of handwritten formula.
}
```

# 

- ### Display mode:

  Drawing in the air.

- ### StyleTransfer mode:

  Style image: Yes..., it's The Starry Night again(@...@)! Here she comes:![The_Starry_Night](./images/theStarryNight.jpg).

1. Whole input Stylized except my body:

   ![Background_and_clothes_stylized](./results/Background_and_clothes_stylized.png)

2. Only clothes stylized(Gif, ahh):

   ![Only_clothes_stylized](./results/Stylization.gif)

- ### Simple Formula Evaluation:

  ​	Since my laptop thinkpad-t450 with i5-5200U and Geforce 940m, I used Lenet... to recognize each single character(coz this is only a very **simple** formula evaluation, I only took some basic operations into account.)

  > The shuffled dataset consists of [MNIST](http://yann.lecun.com/exdb/mnist/) and [handwrittenMathSymbol](https://www.kaggle.com/xainano/handwrittenmathsymbols/). BTW, if you're interested in recognizing a complex mathematic expression, take a look at the MathSymbol dataset, which is from a this kind of competition on Kaggle.

  1. The well-trained Lenet:

     ![Lenet](./images/lenet_training.png)

  2. Then I just split the formula horizontally, just like what I did in the [VehicleLicensePlateRecognition](https://github.com/ZhengPeng7/Vehicle_License_Plate_Recognition).

  3. Afterwards, recognize each single character.

  4. Finally, evaluate the stitched string.