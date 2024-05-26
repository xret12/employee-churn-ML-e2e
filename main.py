from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
from flask_cors import CORS, cross_origin
import flask_monitoringdashboard as dashboard
import pandas as pd
import os

from apps.core.config import Config
from apps.prediction.predict_model import PredictModel
from apps.training.train_model import TrainModel


app = Flask(__name__)
CORS(app)

@app.route('/training', methods=['POST'])
@cross_origin()
def training_route_client():

    try:
        config = Config()
        # get run id
        run_id = config.get_run_id()
        data_path = config.training_data_path
        # initalize TrainModel Object
        trainModel = TrainModel(run_id, data_path)
        trainModel.train_model()


    
        return Response("Training successful!; Run ID: %s" % run_id)
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
    

if __name__ == "__main__":
    app.run()
    host = '0.0.0.0'
    port = 5000
    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()