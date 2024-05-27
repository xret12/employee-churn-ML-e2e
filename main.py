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
dashboard.bind(app)
CORS(app)


@app.route('/', methods=['POST', 'GET'])
def index_page():
    return render_template('index.html')


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


@app.route('/batchprediction', methods=['POST'])
@cross_origin()
def batch_prediction_route_client():
    try:
        config = Config()
        run_id = config.get_run_id()
        data_path = config.prediction_data_path

        # initialize prediction object
        predictModel = PredictModel(run_id, data_path)
        # run prediction
        predictModel.batch_predict_from_model()

        return Response('Prediction successful!; Run ID: %s' % run_id)

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


@app.route('/prediction', methods=['POST'])
@cross_origin()
def single_prediction_route_client():
    try:
        config = Config()
        run_id = config.get_run_id()
        data_path = config.prediction_data_path

        if request.method == 'POST':
            satisfaction_level = request.form['satisfaction_level']
            last_evaluation = request.form['last_evaluation']
            number_project = request.form['number_project']
            average_monthly_hours = request.form['average_monthly_hours']
            time_spend_company = request.form['time_spend_company']
            work_accident = request.form['work_accident']
            promotion_last_5years = request.form['promotion_last_5years']
            salary = request.form['salary']

            data = pd.DataFrame(
                data=[[0, satisfaction_level, last_evaluation, number_project, average_monthly_hours,
                       time_spend_company, work_accident, promotion_last_5years, salary]],
                columns=['empid', 'satisfaction_level', 'last_evaluation', 'number_project',
                         'average_monthly_hours', 'time_spend_company', 'work_accident', 'promotion_last_5years', 'salary']
            )
            # use dictionary to convert specific columns
            convert_dict = {'empid': int,
                            'satisfaction_level': float,
                            'last_evaluation': float,
                            'number_project': int,
                            'average_monthly_hours': int,
                            'time_spend_company': int,
                            'work_accident': int,
                            'promotion_last_5years': int,
                            'salary': object}
            data = data.astype(convert_dict)
            # initialize prediction object
            predictModel = PredictModel(run_id, data_path)
            # run prediction
            output = predictModel.single_predict_from_model(data)

            return Response("Predicted Output is : " + str(output))

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError as k:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


if __name__ == "__main__":
    app.run()
    host = '0.0.0.0'
    port = 5000
    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()
