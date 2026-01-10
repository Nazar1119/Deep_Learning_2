# Hogo Picture

Hot Pot (*Hogo*) - is a dish of soup/stock kept simmering in a pot by a heat source on the table, accompanied by an array of raw meats, vegetables and soy-based foods which diners quickly cook by dipping in broth.

I'm not very familiar with this type of food and don't know how to cook it properly, so I was wondering how much help an assistant could give me with this.
You can see the original image below.
![hogo test image](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_hogo/test_hogo.jpeg)


### Question: 1

- *Question*: Hello, what is this?
- *Output*: Provided only text answer as expected 

#### Chat-example

![hogo chat-example 1](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_hogo/chat_1.png)



### Question: 2

- *Question*: Can you please make me a special reciept for me), and also highlight on image necessary ingredients, because its my first time and i have been mistaken every time in asian food culture `sorry for my english`
- *Output*: Agent provide text answer with image with bounding boxes as exptected

![hogo detected ingredients](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_hogo/hogo_2.jpg)

#### Chat-example

![hogo chat-example 2-1](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_hogo/chat_2_1.png)
![hogo chat-example 2-2](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_hogo/chat_2_2.png)
![hogo chat-example 2-3](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_hogo/chat_2_3.png)



### Question: 3

- *Question*: Provide for me bounding box for all vegetables on this image, use vegetable name as label for each entity
- *Output*: Agent provide only image with boudnding boxes as expected, *i see that model was failed with detected eggs on image, so we request it in question 4*

![hogo detected vegetables](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_hogo/hogo_3.jpeg)

#### Chat-example
![hogo chat-example 3-1](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_hogo/chat_3_1.png)




### Question: 4

- *Question*: Are you see eggs on this image?
- *Output*: Agent provide text answer with bounding boxes, but expected only bounding box. *Model answer that don`t see any eggs on the image, but boudngin box output image contain higlighted eggs on the image.*

![hogo detected eggs](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_hogo/hogo4.jpeg)

#### Chat-example
![hogo chat-example 4-1](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/images/test_hogo/chat_4_1.png)


