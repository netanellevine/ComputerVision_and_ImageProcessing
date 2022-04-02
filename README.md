# Image Processing Assignment 1

This project is the first assignment in the course of Image Processing and Computer Vision.   
In this assignment I'll be doing the following:
1. Read and display an image given.
2. Convert an image between 2 color spaces - [RGB and YIQ][1].
3. Perform [Histogram Equalization][2] on images.
4. Perform [Image Quantization][3].
5. Perform [Gamma correction][4] on Images.

**The files are as follows:**
* **ex1_main.py** - The main file script provided in the assignment
* **myMain.py** - Main file that I created.
* **ex1_utils.py** - The file that contains functions for these actions:
  * Convert an image between RGB and YIQ (both sides)
  * Histogram Equalization
  * Image Quantization  
* **gamma.py** - The file that contains the Gamma correction function
* **Ex1.pdf** - The file with the instructions for this assignment.
* **README.md** - this file
* **images** - folder contains the images given plus the once's I added.
* **output_plots** - folder contains plots of those functions on those images.

_see more about:_  
[Histogram Equalization - Wikipedia](https://en.wikipedia.org/wiki/Histogram_equalization)  
[Image Quantization - Wikipedia](https://en.wikipedia.org/wiki/Quantization_(image_processing))  
[Gamma correction - Wikipedia](https://en.wikipedia.org/wiki/Gamma_correction)
____

## Convert an image between YIQ and RGB

Related functions at **_ex1_utils.py_**:
1. **transformRGB2YIQ**
   1. **param** - _imgRGB_ -> np.ndarray
   2. **return** - _imgYIQ_ -> np.ndarray
2. **transformYIQ2RGB**
   1. **param** - _imgYIQ_ -> np.ndarray
   2. **return** - _imgRGB_ -> np.ndarray

![](output_plots/sunset_color_spaces.png)
![](output_plots/water_bear_3clr_spaces.png)  

[Back to top ][5]
___

## Histogram Equalization

Related functions at **_ex1_utils.py_**:
1. **hsitogramEqualize**
   1. **param** - _imgOrig_ -> np.ndarray
   2. **return** - tuple:
      1. _imgEq_ -> np.ndarray -> new equalized image.
      2. _histOrg_ -> np.ndarray -> histogram of the original image.
      3. _histEQ_ -> np.ndarray -> histogram of the equalized image.

> In the plots below we can see the difference between an image that was dark and therefore the cumulative sum
isn't linear. After the equalization, the image has much more light and from the graph, we can see that the cumulative sum is linear.

![](output_plots/hist_dark_color.png)   
![](output_plots/hist_lighthouse_color.png)
![](output_plots/hist_penguin_gray.png)
![](output_plots/hist_sunset_color.png)

[Back to top ][5]
___

## Image Quantization

Related functions at **_ex1_utils.py_**:
1. **quantizeImage**  quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
   1. **param** - _imgOrig_ -> np.ndarray
   2. **param** - _nQuant_ -> int -> amount of intensities to quantize.
   3. **param** - _nIter_ -> int -> amount of iterations to perform.
   4. **return** - tuple:
      1. _Images_list_ -> List[np.ndarray] -> list of the images after each iteration
      2. _MSE_list_ -> List[float] -> list of the MSE after each iteration
>The plots below show the difference between an image with 256 intensity levels and an image with less. After the quantization, it’s clear that some intensities were merged and the image contains fewer intensity levels. The graph represents the decrease in the MSE value(Mean Square Error) for each iteration of the quantization action, from the original image to the new image. The iterations are finished when the MSE is converged, or we reached the MAX iterations given (lowest of them).

![](output_plots/q4_100iter_uluru_color.png)
![](output_plots/q6_200iter_beach_color.png)
![](output_plots/fewq_uluru_color.png)
![](output_plots/fewq_beach_gray.png)

[Back to top ][5]
___

## Gamma correction

Related functions at **_gamma.py_**:
1. **gammaDisplay** 
   1. **param** - _img_path_ -> str
   2. **return** - _rep_ -> int - 1 for gray 2 for RGB


--- 
      
### Requirements & System preferences

* The system used to implement this project is Mac OS Monterey 12.3.1
* Python version is 3.8.9, using Pycharm.
* Libraries used:
  - [x] Open CV
  - [x] Numpy
  - [x] Matplotlib
  


[1]:#convert-an-image-between-YIQ-and-RGB "RGB and YIQ"
[2]:#Histogram-Equalization "Histogram Equalization"
[3]:#Image-Quantization "Image Quantization"
[4]:#Gamma-correction "Gamma correction"
[5]:#Image-Processing-Assignment-1 "Back to top"
