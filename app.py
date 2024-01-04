from flask import Flask, jsonify, request, render_template
from app_loadrun_models import load_model, run_model

app = Flask(__name__)

###### LOAD MODELS
ax_path = "models/model_ax"
sag_path = "models/model_sag"
cor_path = "models/model_cor"
model_AX, labels_AX, DT_AX, resize_AX = load_model(m_path=ax_path)
model_SAG, labels_SAG, DT_SAG, resize_SAG = load_model(m_path=sag_path)
model_COR, labels_COR, DT_COR, resize_COR = load_model(m_path=cor_path)

###### PREDICT
def predict(model, labels, DT, resize_shape, path):
    if request.method == 'POST':
        
        # Get file path
        req = request.json
        slice_path = req["filepath"]
        # Read file and predict

        # Process prediction
        print('>>>> Predict image...')
        result, uids = run_model(model, labels, DT, resize_shape, slice_path)
        # Format output
        json_output = {}
        json_output['ModelName'] = path        
        json_output['Scores'] = result['scores']
        ### Verion PyDICOM only
        json_output['StudyInstanceUID'] = uids['StudyInstanceUID']
        json_output['SeriesInstanceUID'] = uids['SeriesInstanceUID']
        json_output['SOPInstanceUID'] = uids['SOPInstanceUID']
        ### End version PyDICOM only
        json_predictions = {}
        json_predictions['PredictedClass'] = result['prediction']
        json_predictions['ModelOutputClasses'] = labels
        json_output['Predictions'] = json_predictions
        # Return formatted response
        print('----------- FULL RESPONSE -----------')
        print(json_output)
        return jsonify(json_output), 200

# Method API POST
@app.route('/')
def welcome():
    return "Application is running correctly!"

@app.route('/help')
def render_help():
    return render_template('help.html')

@app.route('/ax', methods=['POST'])
def predict_mr_bp_AX():
    return predict(model_AX, labels_AX, DT_AX, resize_AX, ax_path)

@app.route('/sag', methods=['POST'])
def predict_mr_bp_SAG():
    return predict(model_SAG, labels_SAG, DT_SAG, resize_SAG, sag_path)

@app.route('/cor', methods=['POST'])
def predict_mr_bp_COR():
    return predict(model_COR, labels_COR, DT_COR, resize_COR, cor_path)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("5000"))

