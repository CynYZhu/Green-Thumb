from flask import Flask, render_template, request
from blueprints.recommendations import recommendations 
from blueprints.survey import survey 
from blueprints.cart import cart 
import pandas as pd  
import json     
import os, stat
import shutil


app = Flask(__name__)
app.register_blueprint(recommendations)
app.register_blueprint(survey)
app.register_blueprint(cart)

# samples data source
data_src = './data/full_plant_df.parquet'
df = pd.read_parquet(data_src, engine='pyarrow')

##########################
# Global Paramters
##########################
data = {}
defaults = {}
survey_d = {}

users = {}
uloc={}


##########################
# Helper Functions
##########################
def get_user():
    global users
    ip = request.remote_addr

    if ip in users:
        user = users[ip]
        print('Found User:' + ip)
    else: 
        user = _GreenThumb(ip)
        users[ip] = user
        print('Created User:' + ip)

    return user

def delete_user():
    ip = request.remote_addr    
    path = "/var/www/greenthumb/data/sessions/" + ip

    print('deleting user: ' + ip)
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error %s :  %s" % (path, e.strerror))   

def write_to_file(data):
    ip = request.remote_addr
    path = "/var/www/greenthumb/data/sessions/" + ip
    # checking if the directory folder 
    # exist or not.
    if not os.path.exists(path):        
        # if the folder directory is not present then create it.
        os.umask(0)
        os.makedirs(path, 0o0777)

    # presist the session data for filters so we can read between pages and blueprints
    with open(path + '/userinfo.json', 'w') as f:
        json.dump(data, f)

    os.umask(0)
    os.chmod(path + '/userinfo.json', 0o0777)

##########################
# Flask routes
##########################
@app.route("/")
def home():
    users = get_user()    
    values = {'flowering' : len(df[df['flowering'] == 1]) - 100000
             ,'herbs' : len(df[df['herb'] == 1]) - 90000
             ,'vegetables' : len(df[df['vegetable'] == 1])
             ,'fruiting' : len(df[df['fruit'] != "['none']"])
            }
    return render_template('index.html', data = values)
   
@app.route("/about")
def about():
    users = get_user()    
    return render_template('about.html')
   
@app.route("/privacy")
def privacy():
    users = get_user()    
    return render_template('privacy.html') 

@app.route("/delete_session_data")
def delete_session_data():
    delete_user()
    return render_template('deletedata.html') 

@app.route("/userinfo", methods=['POST'])
def location():
    print('UserInfo Written!')
    d = json.loads(request.values.get('location'))
    
    write_to_file(d)

    return render_template('privacy.html') 
  
@app.route("/plants")
def plantdf():    
    users = get_user()    
    df = pd.read_csv('./data/plants.csv')       
    return df.to_json()

@app.route("/sessions")
def gsessions():
    users = get_user()  
    return {'user'+str(i+1):user for i, user in enumerate(users)}

@app.route("/500")
def get_500():
    return 1/0

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html', error=error) 

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html', error=error) 



## Create Class on incoming IP for sessions state
class _GreenThumb:

    def __init__(self, ip):
        self.ip = ip      

    # Ip Session Connection
    @property
    def ip(self):
        return self.__ip
        
    @ip.setter
    def ip(self, val):
        self.__ip = val    
    
    # Convert to Dictionary/JSON
    def __iter__(self):
        yield 'IP' , self.ip

if __name__ == "__main__":
    app.run()