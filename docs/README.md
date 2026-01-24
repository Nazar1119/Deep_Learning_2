# Documentation
This is the base page for the documentation. Below you can find links to all pages in the documentation.


### Pages
- [Fridge test](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/test_fridge.md) - everyday “fridge photo” scenario to evaluate item recognition + counting and the agent’s ability to return text + annotated bounding boxes; includes a noted failure on citrus counting and localization.  
- [Gotou test](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/test_gotou.md) - sanity-check that the model actually “sees” the image, plus a localization query for a specific text region; shows a routing mismatch where bbox-only was expected but text + bbox was returned. 
- [Hogo test](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/test_hogo.md) - cooking assistant scenario (hot pot): tests describe, recipe generation + ingredient highlighting, and vegetable-only detection; also documents an inconsistency around “eggs” (text says none, but boxes appear).
- [Meerkats test](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/test_meerkats.md) - simple object-part localization: bounding boxes for meerkat heads and a harder counting task (paws) with text + highlighted boxes output.
- [Road test](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/test_road.md) - traffic camera benchmark-style scene: counting trucks, broad scene description + boxes, and reading timestamp from the image.

- [Agentic Logic](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/agentic_logic/README.md) - documentation of the agent’s routing and three execution modes: **TEXT_ONLY**, **BBOX_ONLY**, and **TEXT_AND_BBOX**, including the step-by-step flow for each output type.
