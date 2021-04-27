from google.cloud import speech
import io
import time
import json
#import pandas as pd


def transcribe_file(speech_file):
    """Transcribe the given audio file."""

    client = speech.SpeechClient()

    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    return response
    '''
    print(response)
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(u"Transcript: {}".format(result.alternatives[0].transcript))
    '''

def save_results(results, fn, dest_dir):
    new_fn = '{0}/{1}-result.txt'.format(dest_dir, fn.split('.')[0])
    with open(new_fn, 'w') as f:
        for result in results.results:
            #print(result.transcript)
            f.write(str(result))

def convert_audio_files(fn_list_fn, src_path, dest_dir):
    with open(fn_list_fn, 'r') as audio_fns_f:
        t0 = time.time()
        quota_counter = 0

        for audio_fn in audio_fns_f:
            #Remove .strip when doing whole batch

            audio_fn = audio_fn[:-1].strip('-result.txt').strip('/results/')
            open_fn = "{0}{1}.wav".format(src_path, audio_fn)

            if quota_counter >= 300:
                time_elapsed  = time.time() - t0
                if time_elapsed < 60:
                    time.sleep(60 - time_elapsed)

                    quota_counter = 0
                    t0 = time.time()

            new_fn = '{0}/{1}-result.txt'.format(dest_dir, audio_fn.split('.')[0])
            try:
                with open(new_fn, 'r') as f:
                    pass
            except:
                print(new_fn)
                quota_counter += 1
                save_results(transcribe_file(open_fn), audio_fn, dest_dir)

convert_audio_files('empty_fns.txt', 'CREMA-D/AudioWAV/', 'results')
