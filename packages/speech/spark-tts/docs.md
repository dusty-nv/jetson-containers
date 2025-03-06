### TTS

```bash
python3 /opt/spark-tts/inference.py \
    --pitch "moderate" \
    --speed "moderate" \
    --gender "female" \
    --text "The Quick brown fox jumps over the lazy dog"
```

### Zero-shot Voice Cloning

```bash
python3 /opt/spark-tts/inference.py \
    --prompt_speech_path "/data/audio/dusty.wav" \
    --prompt_text "Hi, this is Dusty. Check, 1, 2, 3. What's the weather going to be tomorrow in Pittsburg? Today is Wendsday, tomorrow is Thursday. I would like to order a large peperroni pizza. Is it going to be cloudy tomorrow?" \
    --speed "very_high" \
    --text "Hi, this is Dusty. I have a quick announcement: SparkTTS is now running smoothly on Jetson! See you down the next rabbit hole!"
```