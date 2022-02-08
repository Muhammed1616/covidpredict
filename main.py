import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from google.cloud import storage


storage_client = storage.Client()
bucket = storage_client.get_bucket('covidpredict1')
blob = bucket.blob('model1.pkl')
blob.download_to_filename('/tmp/model1.pkl')
model = pickle.load(open('/tmp/model1.pkl','rb'))



def api_predict(request):
    if request.method == "GET":
        return "Please send POST Request"
    elif request.method == "POST":
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)

        if(prediction == 0):
            prediction = False
        else:
            prediction = True
        #output = round(prediction[0], 2)
        #return jsonify(prediction)

        return str(prediction)


 