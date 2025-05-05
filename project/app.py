from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Load dataset
df = pd.read_csv('news.csv')

app = Flask(__name__)

@app.route("/")
def home():
    sample_headlines = df['text'].head(10).tolist()
    return render_template("index.html", dataset_headlines=sample_headlines)

@app.route("/get_news/<int:index>")
def get_news(index):
    if 0 <= index < len(df):
        return jsonify(news=df.iloc[index]['text'])
    return jsonify(news="")

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    manual_label = request.form.get('manual_label')

    if manual_label:
        prediction = f"You selected: {manual_label}"
    else:
        data = [news]
        vect = vectorizer.transform(data)
        prediction_result = model.predict(vect)
        prediction = f"Prediction: {prediction_result[0]}"

    sample_headlines = df['text'].head(10).tolist()
    return render_template('index.html', prediction_text=prediction, dataset_headlines=sample_headlines)

if __name__ == '__main__':
    app.run(debug=True)







