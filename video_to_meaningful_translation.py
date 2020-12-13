import argparse
from collections import Counter
import csv
from datetime import datetime
import io
import logging
from os import path, getenv

from google.cloud import speech, storage, translate_v2 as translate
from moviepy.editor import AudioFileClip
import nltk
from pydub import AudioSegment


current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser(description='translate some audio file, it needs to be of lossless format (WAV, AIFF, FLAC..)')
parser.add_argument('--file', help='Video file path to provide')
parser.add_argument('--lang', help='Language of the source audio file', default='zh-CN')
args = parser.parse_args()

VIDEO_FILE = path.join(path.dirname(path.realpath(__file__)), args.file)
audio_clip = AudioFileClip(
    path.join(
        path.dirname(path.realpath(__file__)),
        args.file)
    ).write_audiofile(
        filename=VIDEO_FILE+'.mp3',
        codec='mp3',
        bitrate='320k'
    )

AUDIO_FILE = VIDEO_FILE+'.mp3'
AUDIO_FILE_FLAC = AUDIO_FILE+'.flac'

# Converting mp3 file to FLAC for accuraccy purpose
audio_segment = AudioSegment.from_file(AUDIO_FILE, format='mp3')
audio_flac = audio_segment.export(AUDIO_FILE_FLAC, format='flac')


def upload_audio(source_filename):
    storage_client = storage.Client()
    storage_bucket = storage_client.bucket('vaquita-audio-extracts')
    storage_blob = storage_bucket.blob('audio-extract.flac')

    try:
        storage_blob.upload_from_filename(source_filename)
    except Exception as e:
        logging.error(f"Could not upload audio extract to Google Cloud Storage: {e}")


def get_NNP(keywords: list):
    nouns = list(filter(lambda t: t[1].startswith('NNP'), keywords))
    nouns_list = [noun[0] for noun in nouns]

    return list(Counter(nouns_list).keys())


def write_csv_logfile(video_filename: str, translated_text: str, proper_nouns, current_datetime: str):
    with open('audio_out_results.csv', 'w', newline='') as csv_file:
        fieldnames = ['Video filename', 'Translated text', 'Retrieved Proper Nouns', 'Processed Datetime']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(
            {
                'Video filename': video_filename,
                'Translated text': translated_text,
                'Retrieved Proper Nouns': str(proper_nouns),
                'Processed Datetime': current_datetime
            }
        )


def categorise_words(input_text: str):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    words = nltk.word_tokenize(input_text)
    pos = nltk.pos_tag(words)

    return get_NNP(pos)


recognition = speech.SpeechClient()
audio_upload = upload_audio(AUDIO_FILE_FLAC)
audio = speech.RecognitionAudio(uri='gs://vaquita-audio-extracts/audio-extract.flac') # read the audio file
audio_config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
    audio_channel_count=2,
    language_code=args.lang
)
response = ''

try:
    text_result = recognition.long_running_recognize(
        config=audio_config,
        audio=audio
    )

    operation = text_result.result(timeout=600)

    for result in operation.results:
        response = response + result.alternatives[0].transcript
except Exception as e:
    logging.error(f"Could not retrieve results from Google Cloud Speech to text service: {e}")

if response:
    source_lang = args.lang

    try:
        translator = translate.Client()
        translated = translator.translate(
            response,
            source_language=source_lang,
            target_language='en'
        )

        translated_text = translated['translatedText']
    except Exception as e:
        logging.error(f"Could not retrieve results from Google Cloud translate service: {e}")

    proper_nouns = categorise_words(translated_text)

    write_csv_logfile(args.file, translated_text, proper_nouns, current_datetime)
