# Fridge Picture

Next, we decided to choose an image of a refrigerator to see how useful this system could be for everyday tasks.
You can see original image below.

![fridge test image](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_fridge/test_fridge.jpg)


### Question: 1

- *Question*: What drinks you see on this image?
- *Output*: Agent provide text answer with output image with bounding boxes.

![fridge detected drinks](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_fridge/fridge_1.jpeg)

#### Chat-example

![fridge chat-example 1-1](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_fridge/chat_1_1.png)
![fridge chat-example 1-2](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_fridge/chat_1_2.png)




### Question: 2

- *Question*: How many oranges and lemons you see on this image?
- *Output*: Agent provide text answer with output image with bounding boxes, but expected only text answer. *However, model make mistake in text answer, also failed on output image with bounding boxes.*

![fridge detected lemons and oranges](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_fridge/fridge_2.jpeg)

#### Chat-example

![fridge chat-example 2-1](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_fridge/chat_2_1.png)


### Question: 3

- *Question*: Can you please provide me breakfast meal receipt from my fridge image, and also higlight needable products.
- *Output*: Agent provide text answer with output image with bounding boxes as expected.

![fridge detected breakfast](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_fridge/fridge_3.jpeg)

#### Chat-example

![fridge chat-example 3-1](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_fridge/chat_3_1.png)
![fridge chat-example 3-2](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_fridge/chat_3_2.png)
