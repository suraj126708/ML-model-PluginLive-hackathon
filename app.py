import os
from flask import Flask, request, jsonify, render_template
import whisper
import numpy as np
import tempfile
from datetime import datetime
import soundfile as sf
import librosa
import re
import logging
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Whisper model at startup
try:
    model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {str(e)}")
    model = None

def is_valid_audio(file_path):
    """Validate if the file is a proper audio file."""
    try:
        # First check if it's a valid MP3 file
        with open(file_path, 'rb') as f:
            header = f.read(3)
            if header.startswith(b'ID3') or header.startswith(b'\xff\xfb'):
                # Valid MP3 header found
                return True
        return False
    except Exception as e:
        logger.error(f"Audio validation failed: {str(e)}")
        return False

def detect_repetitions(text: str, segments: list) -> list:
    """Detect word repetitions in speech."""
    words = text.lower().split()
    repetitions = []
    i = 0
    
    while i < len(words) - 1:
        current_word = words[i]
        count = 1
        start_time = None
        end_time = None
        
        j = i + 1
        while j < len(words) and words[j] == current_word:
            count += 1
            j += 1
            
        if count > 1:
            for segment in segments:
                if current_word in segment['text'].lower():
                    if start_time is None:
                        start_time = segment['start']
                    end_time = segment['end']
            
            repetitions.append({
                'word': current_word,
                'count': count,
                'start': start_time,
                'end': end_time
            })
            i = j
        else:
            i += 1
    
    return repetitions

# def analyze_grammar(text: str) -> list:
#     """Basic grammar analysis."""
#     errors = []
#     sentences = re.split('[.!?]+', text)
#
#     for sentence in sentences:
#         words = sentence.strip().lower().split()
#
#         # Check for double negatives
#         negatives = ['not', "n't", 'no', 'never', 'none', 'nothing']
#         neg_count = sum(1 for word in words if any(neg in word for neg in negatives))
#         if neg_count > 1:
#             errors.append({
#                 'type': 'double_negative',
#                 'text': sentence.strip(),
#                 'description': 'Multiple negatives in sentence'
#             })
#
#         # Check subject-verb agreement
#         if len(words) >= 2:
#             singular_subjects = ['i', 'he', 'she', 'it']
#             plural_verbs = ['are', 'were', 'have']
#
#             for i, word in enumerate(words[:-1]):
#                 if word in singular_subjects and words[i+1] in plural_verbs:
#                     errors.append({
#                         'type': 'subject_verb_agreement',
#                         'text': f"{word} {words[i+1]}",
#                         'description': 'Subject-verb agreement error'
#                     })
#
#     return errors

import language_tool_python


def analyze_grammar(text: str) -> list:
    """Enhanced grammar analysis using language_tool."""
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    errors = []

    for match in matches:
        errors.append({
            'type': match.ruleId,
            'text': match.context,
            'description': match.message
        })

    return errors


def analyze_speech(audio_path: str) -> dict:
    """Analyze speech and return comprehensive results."""
    try:
        # Validate audio file
        if not is_valid_audio(audio_path):
            raise ValueError("Invalid audio file format or corrupted file")

        # Load audio file
        logger.debug(f"Loading audio file: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        if model is None:
            raise RuntimeError("Whisper model not properly initialized")

        # Transcribe audio
        logger.debug("Starting transcription")
        result = model.transcribe(
            audio,
            language="en",
            word_timestamps=True,
            initial_prompt="Include hesitations, fillers, and repetitions."
        )
        logger.debug("Transcription completed successfully")

        # Analyze fillers
        filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know']
        fillers = []
        word_count = 0
        
        for segment in result['segments']:
            words = segment['text'].split()
            word_count += len(words)
            
            for word in words:
                word = word.lower().strip('.,!?')
                if word in filler_words:
                    fillers.append({
                        'word': word,
                        'time': segment['start'],
                        'segment_text': segment['text']
                    })

        # Get other analyses
        repetitions = detect_repetitions(result['text'], result['segments'])
        grammar_errors = analyze_grammar(result['text'])
        duration = float(result['segments'][-1]['end']) if result['segments'] else 0
        speech_rate = (word_count / duration * 60) if duration > 0 else 0

        return {
            'status': 'success',
            'basic_metrics': {
                'transcription': result['text'],
                'duration': round(duration, 2),
                'word_count': word_count,
                'speech_rate': round(speech_rate, 2),
                'filler_count': len(fillers),
                'repetition_count': len(repetitions),
                'grammar_error_count': len(grammar_errors)
            },
            'detailed_analysis': {
                'fillers': fillers,
                'repetitions': repetitions,
                'grammar_errors': grammar_errors,
                'segments': result['segments']
            }
        }
    except Exception as e:
        logger.error(f"Error in speech analysis: {str(e)}", exc_info=True)
        raise

@app.route('/')
def index():
    """Render the upload form."""
    return render_template('index.html')

@app.route("/analyze", methods=["POST"])
def analyze_audio():
    """Handle audio file upload and analysis."""
    try:
        # Validate file upload
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400

        file = request.files["audio"]
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # More thorough MP3 validation
        if not file.filename.lower().endswith('.mp3'):
            return jsonify({"error": "Please upload MP3 files only"}), 400

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            try:
                logger.debug(f"Saving uploaded file to temporary location: {temp_file.name}")
                file.save(temp_file.name)
                
                # Validate the audio file
                if not is_valid_audio(temp_file.name):
                    raise ValueError("Invalid or corrupted MP3 file")
                
                # Process the file
                logger.debug("Starting speech analysis")
                results = analyze_speech(temp_file.name)
                
                return jsonify({
                    'status': 'success',
                    'basic_metrics': results['basic_metrics'],
                    'detailed_analysis': results['detailed_analysis']
                })
                
            except Exception as e:
                logger.error(f"Error processing audio file: {str(e)}", exc_info=True)
                return jsonify({"error": str(e)}), 500
            finally:
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logger.error(f"Error deleting temporary file: {str(e)}")

    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(500)
def handle_server_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal Server Error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def handle_not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Resource not found"}), 404

if __name__ == "__main__":
    # Use port 3000 which is more reliable in Codespaces
    port = int(os.environ.get("PORT", 3000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=True
    )