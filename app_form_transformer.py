from flask import Flask, request,render_template
# import torch
# import pandas as pd
# from torchtext.vocab import Vocab, build_vocab_from_iterator
# import csv
# import os
# from janome.tokenizer import Tokenizer
# from werkzeug.utils import secure_filename
from transformers import AutoTokenizer
from functions import pytorch_t





app = Flask(__name__)

@app.route('/form', methods=['GET'])
def get():
    return render_template('form.html')

#postのときの処理 
@app.route('/form', methods=['GET','POST'])
def post():
    sentence = request.form['content']

    generate = pytorch_t.load_nmt(
        '/Users/mayu/flask-1/model/tf_untitled.pt', AutoTokenizer=AutoTokenizer)

    gen = generate(sentence)
    return render_template("result.html",result=gen)




if __name__ == "__main__":
    app.run(debug=True)