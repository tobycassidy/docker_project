FROM python:3.7.2-slim

 

RUN mkdir /code

 

COPY requirements.txt /code/

 
RUN pip install -r /code/requirements.txt

 

COPY nlp_app.py /code/
COPY imdb_reviews_model.sav /
COPY nlp_utils.py /code/
COPY tf_idf_review_vectorizer.sav /


EXPOSE 5000 


CMD ["python","/code/nlp_app.py"]
