# Qwen3-vl Image Agent (Router + Description + Bounding Boxes)

An image-aware agent that routes a user request into one of three flows:
1) text answer only (image description → streamed response),
2) bounding boxes only (object/entity localization → annotated image),
3) both text answer + bounding boxes (dual output).


## Architecture

![Architecture diagram](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/agentic_logic/dark_theme.png)

### Core components
- **Image Router Node**  
  Decides which flow to run based on the user query **only**. The user must clearly describe what they expect in the output, regardless of what is shown in the image.
- **Describe Image**  
  Produces a textual understanding of the image for downstream generation.
- **Extract Entity**  
  Identifies which entity/objects the user is asking about..
- **Generate Output**  
  Generates the final text answer by token streaming.
- **Generate Coordinates**  
  Produces bounding box coordinates for requested entities.
- **Draw Bounding Boxes**  
  Renders the coordinates on the original image.
- **Stream Answer**  
  Streams text tokens/chunks back to the user.
- **Send Image**  
  Returns the annotated image.

## Routing logic (what the router decides)

The router selects exactly one of these modes:

- **TEXT_ONLY**: user only needs a text answer  
  Example: “What’s happening in this photo?”
- **BBOX_ONLY**: user only needs localization/selection on the image  
  Example: “Highlight all the cars” / “Find the logo”
- **TEXT_AND_BBOX**: user needs both explanation + localization  
  Example: “What model is this shoe? Also mark it on the image.”


## Flows

### 1) Text-only flow (Describe → Generate → Stream)
**Goal:** produce a textual answer from image understanding.

Steps:
1. Describe image (vision → text)
2. Generate output with LLM
3. Stream answer to user

### 2) Bounding-box-only flow (Coordinates → Draw → Send image)
**Goal:** return an annotated image with bounding boxes.

Steps:
1. Generate coordinates for requested objects
2. Draw bounding boxes on the image
3. Send image to user

### 3) Text + Bounding boxes (Dual output)
**Goal:** return both a streamed text answer and an annotated image.

Steps:
1. Generate extract entity (what to find in the image)
2. Generate output (text) → stream answer
3. Generate coordinates (for extracted entity)
4. Draw bounding boxes → send image

