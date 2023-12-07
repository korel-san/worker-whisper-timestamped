# whisper_timestamp_worker.py

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
import whisper_timestamped as whisper
import base64
import os
import tempfile
from rp_schema import INPUT_VALIDATIONS

def base64_to_tempfile(base64_file: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))
    return temp_file.name

def whisper_timestamp(job):
    # Get the job input
    job_input = job["input"]

    # Input validation
    # validated_input = validate(job_input, INPUT_VALIDATIONS)
    #
    # if 'errors' in validated_input:
    #     return {"error": validated_input['errors']}

    audio_base64 = job_input["audio"]

    # Decode the audio from base64 and save it as a tempfile
    audio_file = base64_to_tempfile(audio_base64)

    # Load the audio file
    audio = whisper.load_audio(audio_file)

    # Load the model
    model = whisper.load_model("large-v3", device="cuda") # cpu for local

    # Transcribe the audio
    result = whisper.transcribe(model, audio)

    # Delete the audio file
    os.remove(audio_file)

    return result

runpod.serverless.start({"handler": whisper_timestamp})