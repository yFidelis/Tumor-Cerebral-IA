# Start Server
from app.previsor import classificarImagem
import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from app.llm import gerar_explicacao

app = Flask(__name__)

UPLOAD_FOLDER = 'app/static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    dir = os.listdir(app.config['UPLOAD_FOLDER'])
    for file in dir:
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        imagem = request.files.get('file')
        if not imagem:
            return

        filename = secure_filename(imagem.filename)
        imagem_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imagem.save(imagem_path)

        # Classificar a imagem e obter as regiões relevantes
        resultado, heatmap_filename = classificarImagem(imagem_path)

        # Gerar explicação médica usando a LLM
        explicacao = gerar_explicacao(resultado)

        return render_template(
            'result.html',
            result=resultado,
            filename=filename,
            heatmap_filename=heatmap_filename,
            explicacao=explicacao
        )

    return render_template('index.html')

@app.route("/uploadImage", methods=['POST'])
def imageUpload():
    imagem = request.files['image']
    
    filename = secure_filename(imagem.filename)
    imagem_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    imagem.save(imagem_path)

    # Classificar a imagem e obter as regiões relevantes
    resultado, heatmap_filename = classificarImagem(imagem_path)

    # Gerar explicação médica usando a LLM
    explicacao = gerar_explicacao(resultado,)

    return jsonify({
        "resultado": resultado,
        "heatmap_filename": heatmap_filename,
        "explicacao": explicacao
    })

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'), code=301)

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
