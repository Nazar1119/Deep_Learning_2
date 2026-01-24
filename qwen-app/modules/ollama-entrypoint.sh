#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

# Change model that you want to use
echo "👯‍♀️ Download qwen3-vl:32b 👯‍♀️"
ollama pull qwen3-vl:32b
echo "👯‍♀️ Done! 👯‍♀️"

# Wait for Ollama process to finish.
wait $pid


