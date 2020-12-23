from flask import Flask, jsonify, request
import pickle
import sys
import numpy as np
import nlp_utils
import os 

app = Flask(__name__)
tfidf_vect = pickle.load(open(os.path.join('.','tf_idf_review_vectorizer.sav'),'rb'))
model      = pickle.load(open(os.path.join('.','imdb_reviews_model.sav'), 'rb'))


@app.route('/')
def home():
    return jsonify(data='Welcome Home')

@app.route('/review_sentiment')
def review_sentiment():
    review = request.args.get('review','unknown')
    cleaned_review = nlp_utils.process_text(review)
    vectorized_review = tfidf_vect.transform([cleaned_review]).toarray()
    pred = model.predict_proba(vectorized_review)[0, 1]
    print(pred)
    n = np.random.randint(0,4)
    if review == 'unknown':
        results = ["Try again and check your spelling pal",
                    "Didn't quite catch that one",
                    "Are you trying to trip me out",
                    "Have another a go, may have been an input error"]
        return jsonify(response=results[n])
    if pred > 0.70:
        results = ["Wow, you sure are feeling positive about that!",
                   "Awesome, there needs to be more happiness in the world!",
                   "I'll definitely watch this one then, cheers for the heads up",
                   "Smashing, sounds like a blast!"]
        return jsonify(response=results[n], review = review)
    elif pred > 0.5 and pred < 0.7:
        results = ["Sounds like you liked this one",
                   "Pretty good film eh?",
                   "I'm glad you had a good time",
                   "Nice one bruvaaaa!"]
        return jsonify(response=results[n], review = review)
    elif pred < 0.5 and pred > 0.2:
        results = ["Sounds like this one was not your cup of tea",
                   "I recommend you watch another film next time",
                   "Not the best film you've ever seen then?",
                   "Okay, put it in the past then. There's plenty of life ahead."]
        return jsonify(response=results[n],review = review)
    else:
        results = ["Ok, it seems like you're really not that keen on this one..",
                   "Sounds awful.",
                   "Guess you won't be watching this again",
                   "Oh okay...I would advise on you not watching this again?"]
        return jsonify(response=results[n], review = review)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')