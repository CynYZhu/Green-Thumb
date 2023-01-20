from flask import Blueprint, render_template, jsonify, request
import IP2Location as ip2
from datetime import datetime as dt
import pandas as pd
import copy
import joblib
import json
import os, stat
import requests
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder

recommendations = Blueprint('recommendations', __name__)

# samples data source
data_src = './data/full_plant_df.parquet'
df = pd.read_parquet(data_src, engine='pyarrow')

# plant data path (OLD)
plant_path = './data/plant_master_v4.csv'
plant_df = pd.read_csv(plant_path, index_col=0)

envr_path = './data/envr.csv'
envr = pd.read_csv(envr_path, index_col=0, dtype={'zip_code': 'string'})

values = {}

##########################
# Flask routes
##########################
@recommendations.route("/recommendations")
def ui():   
    global values
    values['DEFAULTS'] = defaults()  
    values['SELECTIONS'] = {'attractions': {}
                          , 'water': water()
                          , 'height': height()
                          , 'width': width()
                          , 'temperature': temperature()
                          , 'sun': {}
                          , 'fruit': {}
                          , 'flowering': {}
                          , 'smell': {}
                          , 'vegetable': {}
                          , 'herb': {} # "max":1,"min":0
                          , 'zones': zones()
                          , 'location' : get_location()
                          , 'time_to_plant': values['DEFAULTS']['time_to_plant']
                          , 'selected': ''
                            } 
    values['RECOMMENDATIONS'] = recommedations()

    print('Write to file')
    write_to_file(values)

    return render_template('recommendations.html', values = values)

@recommendations.route("/recommendations", methods=['GET', 'POST'])
def update_values():
    global values    
    values['DEFAULTS'] = defaults()    

    if request.method == 'POST':
        if 'btn_filters_update' in request.form:   
            
            selections = copy.deepcopy(values['DEFAULTS'])
            d = dict(request.form)  
            
            # Attraction
            if 'attractions_0' not in d.keys():
                del selections['attractions'][0]         

            if 'attractions_1' not in d.keys():
                del selections['attractions'][1]                     

            if 'attractions_2' not in d.keys():
                del selections['attractions'][2]  

            if 'attractions_3' not in d.keys():
                del selections['attractions'][3]  

            # Fruit
            if 'fruit_0' not in d.keys():
                del selections['fruit'][0]         

            if 'fruit_1' not in d.keys():
                del selections['fruit'][1]                     

            if 'fruit_2' not in d.keys():
                del selections['fruit'][2]  

            # Sun
            if 'sun_0' not in d.keys():
                del selections['sun'][0]  
                
            if 'sun_1' not in d.keys():
                del selections['sun'][1]  
                
            if 'sun_2' not in d.keys():
                del selections['sun'][2]                        
                
            if 'sun_3' not in d.keys():
                del selections['sun'][3]  
                
            if 'sun_4' not in d.keys():
                del selections['sun'][4]  

            if 'flowering' not in d.keys():
                del selections['flowering']['min']
                del selections['flowering']['max']  

            if 'herb' not in d.keys():
                del selections['herb']['min']
                del selections['herb']['max']  

            if 'smell' not in d.keys():
                del selections['smell']['min']
                del selections['smell']['max']  

            if 'vegetable' not in d.keys():
                del selections['vegetable']['min']
                del selections['vegetable']['max']                


            for k,v in d.items():
                if 'attractions' in k:
                    continue
                if 'fruit' in k:
                    continue
                if 'sun' in k:
                    continue    

                # Height
                if k =='height-max':
                    selections['height']['max'] = v
                    continue
                if k =='height-min':
                    selections['height']['min'] = v
                    continue

                # Width
                if k =='width-max':
                    selections['width']['max'] = v
                    continue
                if k =='width-min':
                    selections['width']['min'] = v
                    continue

                # Temperature
                if k =='temp-max':
                    selections['temperature']['max'] = v
                    continue
                if k =='temp-min':
                    selections['temperature']['min'] = v
                    continue               
                    
                # Flowering
                if k =='flowering' and v == 'on':
                    del selections['flowering']['min']
                    continue

                # Herb
                if k =='herb' and v == 'on':
                    del selections['herb']['min']
                    continue

                # Smell
                if k =='smell' and v == 'on':
                    del selections['smell']['min']
                    continue

                # Vegetable
                if k =='vegetable' and v == 'on':
                    del selections['vegetable']['min']
                    continue                    
                    
                # Time to Plant
                if k =='time_to_plant':
                    r = v.split('/')
                    selections['time_to_plant'] = {'date': v, 'month': r[0], 'day': r[1], 'year': r[2]}
                    
                    continue
                # Zones
                if k =='zones':
                    key = get_key(v, selections['zones'])
                    for i in range(0,12):
                        if key != i:                            
                            del selections[k][i]
                    continue

                # ZipCode
                if k =='location':
                    selections[k]['postal'] = v
                    continue  

                # Water
                if k =='water':
                    key = get_key(v, selections['water'])
                    for i in range(0,4):
                        if key != i:                            
                            del selections['water'][i]
                    continue           

                if k == 'selected':
                    selections['selected'] = v

            values['SELECTIONS'] = selections

        else:
            values['NOFORM'] = 1
    else:
        values['GET'] = 1

    values['RECOMMENDATIONS'] = recommedations()
    write_to_file(values)    

    return render_template('recommendations.html', values = values)
    
@recommendations.route("/recommendations/data")
def json_recommendations():
    return values

def defaults():  
    v = dt.now().strftime('%m/%d/%Y')
    r = v.split('/')

    d = {'attractions': attractions()
       , 'water': water()
       , 'height': height()
       , 'width': width()
       , 'temperature': temperature()
       , 'sun': sun()
       , 'fruit': fruit()
       , 'flowering': flowering()
       , 'smell': smell()
       , 'vegetable': vegetable()
       , 'herb': herb()
       , 'zones': zones()
       , 'location' : get_location()
       , 'time_to_plant' : {'date': v, 'month': r[0], 'day': r[1], 'year': r[2]}
       , 'selected': ''
       } 
    return d
   

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
    with open(path + '/recommendations.json', 'w') as f:
        json.dump(data, f)

    os.umask(0)
    os.chmod(path + '/recommendations.json', 0o0777)
    print("4")

##########################
# Helper Functions
##########################
def to_camel_case(text):
    s = text.replace("-", " ").replace("_", " ").split()
    s = [t.capitalize() for t in s]
    return ' '.join(s)

# Min
def _min(l):
    a = None
    for n in l:
        if (a is None or n < a):
            a = n
            
    return a    
    
# Max
def _max(l):
    a = None
    for n in l:
        if (a is None or n > a):
            a = n
            
    return a

# Unique list
def unique(values):
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    
    return unique

# function to return key for any value
def get_key(val, d):
    for key, value in d.items():
        if val == value:
            return key
 
    return "key doesn't exist"    

###########################################
# FILTERS: Map all filter data from source
###########################################

def attractions():
    # MultiSelect Checkbox
    #{0: 'butterflies', 1: 'none', 2: 'hummingbirds', 3: 'birds'}
    return {i: v for i, v in enumerate(unique([item for sublist in df['attracts'].unique() for item in sublist.strip('][\'').replace("', '", ', ').split(', ')]))}

def water():   
    # DropDown
    #{'dry', 'medium', 'none', 'wet'}
    return {i: v for i, v in enumerate(unique([item for sublist in df['water'].unique() for item in sublist.strip('][\'').replace("', '", ', ').split(', ')]))}
      
def height(): 
    # Slider Range 
    #{'MIN': 0, 'MAX': 180}
    l = set([item for sublist in df['height'].unique() for item in sublist.strip('][\'').replace("', '", ', ').split(', ')])
    l = [int(i) for i in l]
    return {'min' : _min(l), 'max' : _max(l)}

def width():   
    # Slider Range 
    #{'MIN': 0, 'MAX': 79}
    l = set([item for sublist in df['width'].unique() for item in sublist.strip('][\'').replace("', '", ', ').split(', ')])
    l = [int(i) for i in l]
    return {'min' : _min(l), 'max' : _max(l)}      

def temperature():   
    # Slider Range 
    #{'MIN': 0, 'MAX': 104}
    l = set([item for sublist in df['temperature'].unique() for item in sublist.strip('][\'').replace("', '", ', ').split(', ')])
    l = [int(i) for i in l]
    return {'min' : _min(l), 'max' : _max(l)}    
    
def sun():  
    # MultiSelect Checkbox
    #{'full shade', 'full sun', 'none', 'part shade', 'part sun'}
    return {i: v for i, v in enumerate(unique([item for sublist in df['sun'].unique() for item in sublist.strip('][\'').replace("', '", ', ').split(', ')]))}
     
def fruit():  
    # MultiSelect Checkbox
    # {'edible', 'none', 'showy'}
    return {i: v for i, v in enumerate(unique([item for sublist in df['fruit'].unique() for item in sublist.strip('][\'').replace("', '", ', ').split(', ')]))}
     
def flowering():  
    # CheckBox 
    #{'MIN': 0, 'MAX': 1}
    l = set(df['flowering'].unique())
    return {'min' : int(_min(l)), 'max' : int(_max(l))} 

def smell():  
    # CheckBox 
    #{'MIN': 0, 'MAX': 1}
    l = list(df['smell'].unique())
    return {'min' : int(_min(l)), 'max' : int(_max(l))}  

def vegetable():  
    # CheckBox 
    #{'MIN': 0, 'MAX': 1}
    l = list(df['vegetable'].unique())
    return {'min' : int(_min(l)), 'max' : int(_max(l))} 

def herb():  
    # CheckBox 
    #{'MIN': 0, 'MAX': 1}
    l = list(df['herb'].unique())
    return {'min' : int(_min(l)), 'max' : int(_max(l))} 

def zones():  
    # MultiSelect Checkbox
    #{'0', '1', '10', '11', '2', '3', '4', '5', '6', '7', '8', '9'}
    return {i: v for i, v in enumerate(unique([item for sublist in df['zones'].unique() for item in sublist.strip('][\'').replace("', '", ', ').split(', ')]))}      

def sun_keys(keys):  
    # if label CheckBox (flowering, smell, herb, vegetable)
    #{'MIN': 0, 'MAX': 1} 
    
    sun_path = './data/sun.parquet'
    sun = pd.read_parquet(sun_path, engine='pyarrow')

    vals = list(values['SELECTIONS']['sun'].values())

    if sun.columns.to_list().sort() == (vals + ['id']).sort():
        return keys
    
    r = []
    for v in vals:
        q = list(sun[(sun[v] == 1)]['id'])
        r = r + q

    return r

def get_time_to_plant():    

    if 'SELECTIONS' in values.keys():
        if 'time_to_plant' in values['SELECTIONS'].keys():
            v = values['SELECTIONS']['time_to_plant']
            r = v.split('/') 
            return {'date': v, 'month': r[0], 'day': r[1], 'year': r[2]}

    v = dt.now()
    return {'date': dt.now().strftime('%m/%d/%Y'), 'month':str(v.month), 'day': str(v.day), 'year': str(v.year)}
   
def get_location():

    ip = request.remote_addr
    path = "/var/www/greenthumb/data/sessions/" + ip 
    
    #Opening JSON file for filters on recommedations page
    location = {}
    with open(path + '/userinfo.json', 'r') as f:
        location = json.load(f)

    return location   

def get_survey():

    ip = request.remote_addr
    path = "/var/www/greenthumb/data/sessions/" + ip 
    
    #Opening JSON file for filters on recommedations page
    survey = {}

    try:
        with open(path + '/survey.json', 'r') as f:
            survey = json.load(f)
    except:
        print('No Survey.json')
        survey['Summary'] = {}

    return survey

##########################
# Modeling
##########################
## current filters that work for the master_v3
def jdata():
    return {
    'type': 
             {'1': 'flowering'},  #, , '2': 'greens', '3': 'hybrid'
    'height': 
             {'max': 10, 
              'min': 0}, 
    'width':
             {'max': 30, 
              'min': 0},
    'smell': 
             {'min': 1}, 
    'showy': 
             {'max': 0},
    'maintenance': 
             {'1': 'high'},
    'zipcode': 
             {'ip': '67.170.250.166',
              'zip': '83703'},
    'month': {'0': 3}
}
#return {
#    'type': {'1': 'flowering'},  #, , '2': 'greens', '3': 'hybrid'
#    'height': {
#        'max': 10 # values['SELECTIONS']['height']['max'], 
#        ,'min': 0  # values['SELECTIONS']['height']['min']
#    }, 
#    'width':{'max': 30 # values['SELECTIONS']['width']['max']
#            ,'min': 0  # values['SELECTIONS']['width']['min']
#    },
#    'smell': {'min': 1}, 
#    'showy': {'max': 0},
#    'maintenance': {'1': 'high'},
#    'zipcode': {'ip': values['SELECTIONS']['location']['ip'],
#                'zip': values['SELECTIONS']['location']['postal']},
#    'month': {'0': 3}   # values['SELECTIONS']['time_to_plant']['month']}
#}
def envr_features():
    global envr

    j = jdata()
    
    # get closest zip code
    lst = [int(e) for e in envr['zip_code'].unique()]    
    K = int(j['zipcode']['zip'])
    zipcode = str(lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))])

    # get closest month
    month = [m for _, m in j['month'].items()][-1]
    
    def temp_bucket(x):
        if x <= 55:
            o = 'low'
        elif x in range(56, 66):
            o = 'cool'
        elif x in range(66, 75):
            o = 'mild'
        elif x in range(75, 85):
            o = 'warm'
        elif x in range(85, 200):
            o = 'hot'
        return o

    def sun_bucket(x):
        '''
        Full sun - 6 or more hours of direct sun per day
        Part sun - 4 to 6 hours of direct sun per day, including some afternoon sun
        Part shade - 4 to 6 hours of direct sun per day, mostly before midday
        Full shade - less than 4 hours of direct sun per day
        '''
        x = round(x)
        if x <= 4:
            out = 'full shade'
        elif x in range(4, 7):
            out = 'part sun'
        elif x in range(6, 26):
            out = 'full sun'
        return out
        
    loc_info = envr[(envr['zip_code'] == zipcode) & (envr['Month'] == month)].to_dict()
    temp_cat = temp_bucket(list(loc_info['tmin'].values())[0])
    zone = list(loc_info['zone'].values())[0]
    sun_cat = sun_bucket(list(loc_info['GHI_per_day'].values())[0])
    
    return zone, sun_cat, temp_cat
# read plant data
def subset_plant():
    
    """ input: 
            plant_path: path to plant master data
            filter_params: filters ingested from front-end json and environment features
        output: 
            subset of plants
    """
    global plant_df
    try:
        # variables
        j = jdata()
        params = []
        for _, typ in j['type'].items():
            type_col = typ
        for _, sm in j['smell'].items():
            smell = sm
        for _, sw in j['showy'].items():
            showy = sw
        for _, mt in j['maintenance'].items():
            if mt == 'high':
                maintenance = ['low', 'medium', 'high', 'none']
            elif mt == 'medium':
                maintenance = ['low', 'medium', 'none']
            else: 
                maintenance = ['low', 'none']

        height_max = j['height']['max']
        height_min = j['height']['min']
        width_max = j['width']['max']
        width_min = j['width']['min']
        
        # unpack filters
        type_col, smell, showy, maintenance, height_max, height_min, width_max, width_min 
        zone, sun_cat, temp_cat = envr_features()

        sub = plant_df[(plant_df[type_col] == 1) 
                    & (plant_df['smell']== smell) 
                    & (plant_df['showy']== showy) 
                    & (plant_df['sun']== sun_cat) 
                    & (plant_df['maintenance'].isin(maintenance))
                    & ((plant_df['height']<=height_max) & (plant_df['height']>=height_min)) 
                    & ((plant_df['width']<=width_max) & (plant_df['height']>=width_min)) 
                    & (plant_df['zones'].isin([zone-1, zone, zone+1]))
                    & (plant_df['temp_bucket']== temp_cat)]

        # remove the filter columns
        cols_to_drop = ['smell', 'showy','sun', 'maintenance', 'height', 'width', 'zones', 'temp_bucket', type_col]
        sub.drop(columns=cols_to_drop, inplace=True)
    except:
        print('error in sub set plant')
        sub = plant_df

    return sub
    
def pca(df):
    """ function: one-hot encoded catgorical features concatonated with binary feature
        input: subset of plant
        output: dimentionality reduced pca data
    """
    
    cat_cols = ['attracts', 'water', 'special_feature', 'propagation',
           'problem_solvers', 'm_type', 'bloom_season']

    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    encoded_columns = categorical_preprocessor.fit_transform(df[cat_cols])
    processed_data = encoded_columns.todense()

    while True:
        try:
            pca = PCA(n_components=processed_data.shape[1],random_state=42)
            pca_model = pca.fit(processed_data)

        except ValueError: 
            pca_data = processed_data
            break

        else:
            ratio = pca_model.explained_variance_ratio_
            cum_sum_eigenvalues = np.cumsum(ratio)

            # select 95% cutoff
            cutoff = np.where(cum_sum_eigenvalues>0.95)[0][0]
            pca = PCA(n_components=cutoff,random_state=42)
            pca_data = pca.fit_transform(processed_data)
            break

    bin_cols = [col for col in df.columns if col not in cat_cols or not 'common']
    target = df.pop('common').tolist()
    bin_cols.remove('common')
    bin_train = np.array(df[bin_cols])

    features = np.concatenate((bin_train, pca_data),axis=1) 
    
    return features, target
def recommendation(features, target, plant, df):
    ''' 
    funtion: recommendation engine
    input:
        features: features for cosine similarity calc.
        target: names of plants
        plant: randomly selected item, used to find similarlity for
        df: subset of plant dataframe
        NUM_SIMILAR: num of recommendation to return
    output:
        df of top n similar items
    '''
    
    item_idx = target.index(plant)
    query_vec = features[item_idx][:]

    u = query_vec
    V = features
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    u = u / np.linalg.norm(u)
    scores = u.dot(V.T)
    df['cosine'] = scores

    df['target']= target
    score_df = pd.DataFrame(df.groupby('target')['cosine'].mean().sort_values(ascending=False))

    return score_df

def load():    
    return joblib.load("recommeder.pkl")

def save():
    joblib.dump(model, "recommeder.pkl")

def predict(arr):
    from_jb.predict(arr)   


# Randomly fill the survey from current data states and the source
def recommedations():

    # Get ids with pics
    keys = []
    p = '/var/www/greenthumb/static/assets/images/plants/'
    count = 12

    # filter keys
    data = df['id']
    #data = sun_keys(data)
    #data = get_data(data, 'flowering')
    #data = get_data(data, 'smell')
    #data = get_data(data, 'vegetable')
    #data = get_data(data, 'herb')

    # get recommendation keys
    mdf = subset_plant()
    survey = get_survey()
    

    if survey == {}: # then randomize 
        # mock user selection
        print('Error psycho one detected!')
        USER_PREF_PLANT = mdf.common.sample(1).values[0]        
    else:
        try:
            if values['SELECTIONS']['selected'] == '':
                for c in survey['Summary'].keys():
                    if c != 'down':
                        key = list(survey['Summary'][c]['items'].keys())[0]
                        values['SELECTIONS']['selected'] = key
                        break
            else:
                key = int(values['SELECTIONS']['selected'])

            USER_PREF_PLANT = df.loc[df['id'] == int(key), 'common'].values[0]    
        except:
            print('Error psycho two detected!')
            USER_PREF_PLANT = mdf.common.sample(1).values[0]  
    
    rkeys = {}
    try:
        features, target = pca(mdf)
        sim_items = recommendation(features, target, USER_PREF_PLANT, mdf)
        for i, (r,c) in enumerate(zip(sim_items.index[:count+1].to_list(), sim_items[:count+1].values.tolist())):
            rkeys[i] = {
                    'key': df.loc[df['common'] == r, 'id'].values[0]
                    , 'common': r
                    , 'cosine' : "{:.2%}".format(c[0])
                    }
        

        if rkeys != []: data = df['id'][[v['key'] for k, v in rkeys.items()]]

    except:
        print('Error getting plant rec')
         # cut idx that have no pictures
        for idx in data.tolist():
            f =  str(idx) + '.jpg'
            if os.path.exists(os.path.join(p, f)):
                keys.append(idx)          

        random.shuffle(keys)

        for i, k in enumerate(keys[:count+1]):
            c = [0]
            rkeys[i] = {
                      'key': k
                    , 'common': df.loc[df['id'] == k, 'common'].values[0]
                    , 'cosine' : "{:.2%}".format(c[0])
                    }
        

    # cut idx that have no pictures
    for idx in data.tolist():
        f =  str(idx) + '.jpg'
        path = os.path.join(p, f)
        # if path has picture use it
        if os.path.exists(path):
            # if picture is big enough its real
            if os.stat(path).st_size > 10000:
                keys.append(idx)     

    random.shuffle(keys)

    recs = {}
    # Head
    recs['head'] = {
        'count': len(keys)
      , 'survey': survey
    }
    
    # Body
    recs['body'] = {}
    mlen = df.iloc[keys]['genus'].str.len().max()
    for i, idx in enumerate(keys[0:count]): #// getting an (i,d) in the html
        genus = df[df['id'] == int(idx)]['genus'].to_string(index=False) 
        genus = (genus[0:27] + '...') if (len(genus) > 27) else genus        
            
        native_region = df[df['id'] == int(idx)]['native_region'].to_string(index=False) 
        try:
            native_region = native_region.strip('[].').replace("'","").replace(",","").split(" ")[0].replace('none','')
            if native_region not in ['europe', 'china', 'india', 'polynesia'] : native_region = ''
        except:
            pass

        recs['body'][i] = {'idx' : str(idx)
                          ,'common': to_camel_case(df[df['id'] == int(idx)]['common'].to_string(index=False))
                          ,'cosine' : [v['cosine'] for k, v in rkeys.items()][i] if rkeys != {} else ''
                          ,'genus' : genus
                          ,'full_genus' : df[df['id'] == int(idx)]['genus'].to_string(index=False) 
                          ,'native_region' : native_region}

    # random select options to show
    return recs   