# HCI_lite
> A lite human computer interaction system, mainly dependent on you hand in your webcam.
> Author: Peng Zheng.
>
> Duration: 
>
> |                           Progress                           | Start Date | Deadline  |
> | :----------------------------------------------------------: | :--------: | :-------: |
> | Building basic video flow frame skeleton, including controller tracking, controller tailing... |  5/9/2018  | 5/9/2018  |
> |                     Appending top menu.                      | 5/10/2018  | 5/10/2018 |
> |        Inserting real time image_style_transfer mode.        | 5/10/2018  | 5/11/2018 |
> |  Implementing clothes region extraction for styleTransfer.   | 5/11/2018  | 5/12/2018 |
> | Implementing formula evaluation: integerate my own dataset for training Lenet to do single character recognition, split the handwritten expression, then evaluate it. | 5/13/2018  | 5/14/2018 |
> |              Finishing sunglasses wearing mode.              | 5/14/2018  | 5/15/2018 |
> |      Refining my documents and improving the stability.      | 5/13/2018  |    ...    |
> |                    __Development SumUp__                     | 5/10/2018  | 5/15/2018 |



## Dependencies:

    OpenCV==3.4.0
    numpy==1.14.3
    matplotlib==2.2.2
    tensorflow-gpu==1.8.0    # CUDA=9, CUDNN=7.
    scikit-learn==0.19.1
    Keras==2.1.6
    face-recognition==1.2.2


## Outline:
![outline](./images/outline.svg)

# Mode:

- ### Guide:

```python3
video mode setting: {
"display": Default value, of which ink would fade, with tracking effect,
"styleTransfer": Stylize the whole input from webcam or only your clothes,
"calc": Do math evaluation from formula you've written on the screen,
      "evaluation": evaluate the result of handwritten formula,
"glass": Help you wear a pair of glasses,
}
```

# 

- ### Display mode:

  __Algorithms__: No one deserves mention.

  Drawing in the air.

- ### StyleTransfer mode:

  Style image: Yes..., it's The Starry Night again(@...@)! Here she comes:![The_Starry_Night](./images/theStarryNight.jpg).

1. Whole input is stylized except my body:

   __Algorithms:__ Skin Detection + Basic Morphology operations, etc.

   ![Background_and_clothes_stylized](./images/Background_and_clothes_stylized.png)

2. Only clothes stylized(Gif, ahh):

   __Algorithms:__ 

    	1. Background Substraction (LSBP) / Grab Cut.
    	2. Skin Detection

   ![Only_clothes_stylized](./images/Stylization.gif)

- ### Simple Formula Evaluation:

  ​	Concerning my laptop thinkpad-t450 with i5-5200U and Geforce 940m... I used Lenet. to recognize each single character(coz this is only a very **simple** formula evaluation, I only took some basic operations into account.)

  > The shuffled dataset consists of [MNIST](http://yann.lecun.com/exdb/mnist/) and [handwrittenMathSymbol](https://www.kaggle.com/xainano/handwrittenmathsymbols/). BTW, if you're interested in recognizing a complex mathematic expression, take a look at the MathSymbol dataset, which is from a this kind of competition on Kaggle.

  1. The well-trained Lenet:

     ![Lenet](./images/lenet_training.jpg)

  2. Then I just split the formula horizontally, just like what I did in the [VehicleLicensePlateRecognition](https://github.com/ZhengPeng7/Vehicle_License_Plate_Recognition).

  3. Afterwards, recognize each single character.

  4. Finally, evaluate the stitched string.

- ### Glass mode:

  __Algorithms:__ [Haarcascade](https://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) for eye detection.

  ![glass_modE](./images/glass_modE.gif)

- ### AR for Time Tower

  __Algorithms:__ [3d_calibration](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#solvepnp).
