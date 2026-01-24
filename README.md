# Qwen3-vl Object Detection (Academic Project)
A web application with custom agent logic for testing a models(*Qwen3-vl*) ability to detect objects in an image.

### Project structure
- `docs` - directory where you can find experimentation and other documentation pages [docs page router](https://github.com/sidjik/obj-det-chat-qwen3-vl/blob/master/docs/README.md).
- `modules` - directory where you can find source-code for this project.

### Quick Setup
*Make sure that ollama successfully runing on localhost:11434, it is very important that ollama works on the `0.0.0.0` interface. This is necessary because containers need to have access to ollama. If you want use ollama like conatiner in compose, uncomment ollama service in compose file, and also uncomment 83 line in compose file.*

##### Run Docker Compose

1. Change directory to the `modules` folder.
2. Start the services:

**Unix/macOS**

```sh
docker compose up
```

**Windows (Command Prompt)**

```cmd
docker-compose up
```

---

That’s it — your service is now running.


