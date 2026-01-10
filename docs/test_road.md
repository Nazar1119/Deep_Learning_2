# Road-Camera picture

We chose the next image as a classic example for computer vision tasks—namely, a picture from a traffic camera. This model has proven to be quite useful for tasks involving such images.
You can see original image below.

![road test image](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_road/test_road.jpg)


### Question: 1

- *Question*: How many trucks are you see on this image?
- *Output*: Agent provide text answer and output image with bounding boxes. *However, model does not recognize a long way entity*.

![road detected trucks](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_road/road_1.jpg)

#### Chat-example
![road chat-example 1-1](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_road/chat_1_1.png)
![road chat-example 1-2](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_road/chat_1_2.png)


### Question: 2

- *Question*: Can you describe what you see on the image, and also provide bounding boxes for thing that you see on the image.
- *Output*: Agent provide text answer and output image with bounding boxes, as expected. *However, model does not recognize one car near one of truck.*

![road detected entity](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_road/road_2.jpg)

#### Chat-example

![road chat-example 2-1](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_road/chat_2_1.png)
![road chat-example 2-2](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_road/chat_2_2.png)
![road chat-example 2-3](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_road/chat_2_3.png)



### Question: 3

- *Question*: What date and time i have on the left top corner of image?
- *Output*: Agent provide text answer and output image with bounding boxes.

![road detected date](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_road/road_3.jpg)

#### Chat-example

![road chat-example 3-1](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_road/chat_3_1.png)
![road chat-example 3-2](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_road/chat_3_2.png)
![road chat-example 3-3](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_road/chat_3_3.png)


### Question: 4

- *Question*: Can you answer on what road this image was shot?
- *Output*: Agent provide only text answer that expected.


#### Chat-example
![road chat-example 4-1](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_road/chat_4_1.png)
