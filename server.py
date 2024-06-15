from flask import Flask, request, jsonify, render_template, Response
from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile
from io import BytesIO

app = Flask(__name__, static_folder='static')

# Load the model and tokenizer globally so they can be reused
model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-audio', methods=['POST'])
def text_to_speech():
    try:
        # Get the text from the request body
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({"success": False,'errors': [{"code": "BAD_REQUEST", "message": "no text provided"}]}), 400

        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt")

        # Generate the audio waveform with the model
        with torch.no_grad():
            output = model(**inputs).waveform

        # Convert the waveform to a numpy array
        output = output.squeeze().cpu().numpy()

        # Save the audio to a BytesIO object
        byte_io = BytesIO()
        scipy.io.wavfile.write(byte_io, rate=model.config.sampling_rate, data=output)
        byte_io.seek(0)

        # Return the audio content with 'Content-Type': 'audio/mpeg'
        return Response(byte_io.read(), mimetype='audio/mpeg')

    except Exception as e:
        return jsonify({"success": False,'errors': [{"code": "INTERNAL_SERVER_ERROR", "message": str(e)}]}), 500

if __name__ == '__main__':
    app.run(debug=True)
