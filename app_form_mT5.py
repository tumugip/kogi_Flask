from flask import Flask, request,render_template
from transformers import (
    MT5ForConditionalGeneration,AutoTokenizer
)
from functions import pytorch_t 

app = Flask(__name__)

@app.route('/form', methods=['GET'])
def get():
    return render_template('form.html')

#postのときの処理 
@app.route('/form', methods=['GET','POST'])
def post():
    sentence = request.form['content']

    model = MT5ForConditionalGeneration.from_pretrained('/Users/mayu/flask-1/model/kogi_6_0603_result_util')

    tokenizer = AutoTokenizer.from_pretrained('/Users/mayu/flask-1/model/kogi_6_0603_result_util')

    generate = pytorch_t.make_generate(model, tokenizer)

    gen = generate(sentence)
    
    return render_template("result.html",result=gen)



if __name__ == "__main__":
    app.run(debug=True)