from flask import Flask, render_template, request, url_for, Markup, jsonify,redirect
import pickle
import requests
import json
from newspaper import Article
import pytesseract
from PIL import Image
import io
import urllib

app = Flask(__name__)
pickle_in_text = open('model_text.pickle','rb')
pac_text = pickle.load(pickle_in_text)

pickle_in_title = open('model_title.pickle','rb')
pac_title = pickle.load(pickle_in_title)

tfid_text = open('tfid_text.pickle','rb')
tfidf_vectorizer_text = pickle.load(tfid_text)

tfid_title = open('tfid_title.pickle','rb')
tfidf_vectorizer_title = pickle.load(tfid_title)

@app.route('/')
def home():
 	return render_template("index.html")
 	
@app.route('/newscheck',methods=["GET","POST"])
def newscheck():	
	abc = request.args.get('news')	
	input_data = [abc.rstrip()]
	# transforming input
	tfidf_req = tfidf_vectorizer_text.transform(input_data)
	# predicting the input
	y_pred_text = pac_text.predict(tfidf_req)
	return jsonify(result = y_pred_text[0])

@app.route("/article_url", methods=["GET","POST"])
def responses():
    if request.method == 'POST':
        url = request.get_data(as_text=True)[5:]
        url = urllib.parse.unquote(url)
        article = Article(str(url))
        article.download()
        article.parse()
        news_title = tfidf_vectorizer_title.transform(article.title)
        news_text = tfidf_vectorizer_text.transform(article.text)
        y_pred_text = pac_text.predict(news_text)
        y_pred_title = pac_title.predict(news_title)
        if (y_pred_text[0] == "REAL" and y_pred_title[0] == "REAL"):
            return jsonify(result = y_pred_text[0])
        elif (y_pred_text[0] == "REAL"):
            return jsonify(result = y_pred_text[0])
        else:
            return jsonify(result = "FAKE")
    return render_template("check_fake_url.html")
    

@app.route('/scan', methods=['GET', 'POST'])
def scan_file():
    if request.method == 'POST':
        image_data = request.files['file'].read()

        scanned_text = pytesseract.image_to_string(Image.open(io.BytesIO(image_data)))
        tfidf_req = tfidf_vectorizer_text.transform(scanned_text)
        y_pred_text = pac_text.predict(tfidf_req)
        return jsonify(result = y_pred_text[0])
    return render_template("check_fake_image.html")
    


if __name__=='__main__':
    app.run(debug=True)