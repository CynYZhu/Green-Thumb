from flask import Flask, render_template, request
import urllib
import json
import os
 
app = Flask(__name__)

##########################
# Flask routes
##########################
# render index.html home page
@app.route("/")
def home():
   # return "<h1>Hello World</h1>"
   return render_template('index.html')
   
@app.route("/about")
def about():
   return render_template('about.html')
   
@app.route("/privacy")
def privacy():
   return render_template('privacy.html')

@app.route("/", methods=['GET', 'POST'])
def my_form_post():
    if request.method == 'POST':
        text = request.form['u']
        processed_text = text
        response = getData(processed_text)

    return response

def getData(input_text):
    data = {
        "Inputs": {
            "WebServiceInput0":
            [
                {
                        'CODE_med': "0",
                        'DESCRIPTION_med': input_text,
                        'CODE': "0",
                        'DESCRIPTION': "0",
                },
            ],
        },
        "GlobalParameters":  {
        }
    }
    body = str.encode(json.dumps(data))
   #testing
    url = ''
    api_key = '' # Replace this with the API key for the web service
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read().decode('utf-8')
        obj = json.loads(result)
        
        return obj['Results']['WebServiceOutput0'][0]['Scored Labels']

    except urllib.error.HTTPError as error:
        return "The request failed with status code: " + str(error.code)

   
@app.errorhandler(500)
def internal_error(error):
    return "500 error"

@app.errorhandler(404)
def not_found(error):

    return "404 error",404

if __name__ == "__main__":
    app.run()
