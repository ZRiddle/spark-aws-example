from flask import Flask, request, json
import boto3
import pickle

BUCKET_NAME = 'sagemaker-2018-09-04'
MODEL_FILE_NAME = 'iris.pkl'
REGION = 'us-west-2'

app = Flask(__name__)
S3 = boto3.client('s3', region_name=REGION)


def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return helper


@memoize
def load_model(key):
    response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
    model_str = response['Body'].read()

    model = pickle.loads(model_str)

    return model


@app.route('/', methods=['POST'])
def index():    
    # Parse request body for model input 
    body_dict = request.get_json(silent=True)    
    data = body_dict['data']     
    
    # Load model
    model = load_model(MODEL_FILE_NAME)
	# Make prediction 
    prediction = model.predict(data).tolist()
	# Respond with prediction result
    result = {'prediction': prediction}    
    
    try:
    	return json.dumps(result)
    except:
    	return "Error"


if __name__ == '__main__':    
    # listen on all IPs 
    app.run(host='0.0.0.0')
