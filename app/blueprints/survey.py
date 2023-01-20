from flask import Blueprint, render_template, jsonify, request
import pyarrow.parquet as pq
from pandasql import sqldf
import pandas as pd
import random
import json
import os
import copy

survey = Blueprint('survey', __name__)

# samples data source
data_src = './data/full_plant_df.parquet'
df = pd.read_parquet(data_src, engine='pyarrow')
plant_path = './data/plant_master_v4.csv'
plant_df = pd.read_csv(plant_path, index_col=0)

selections = {}

##########################
# Flask routes
##########################
@survey.route("/survey", methods=['GET', 'POST'])
def ui():
    global selections, df   

    cnt = 0
    if len(selections) == 0:
        cnt = 1
    else:
        cnt = len(selections['Instances']) + 1

    if request.method == 'POST':
        if 'btn_refresh_survey' in request.form:     
            d = dict(request.form)       
                  

            # Apend instance to instances
            if 'Instances' in selections.keys():          
                selections['Instances'][cnt] = {}
            else:
                selections['Instances'] = {cnt : {}}

            # Append summary    
            if cnt == 1: 
                selections['Summary'] = {}

            for flabel in d.keys():

                if flabel == 'btn_refresh_survey' : continue

                kv = flabel.split('_')
                key, state = kv[0], kv[1]
                label = df.loc[df['id'] == int(key), 'genus'].values[0]
                clabel = df.loc[df['id'] == int(key), 'common'].values[0]

                if state not in selections['Instances'][cnt].keys(): 
                    selections['Instances'][cnt][state] = {}
                    selections['Instances'][cnt][state]['items'] = {} 


                if state not in selections['Summary'].keys(): 
                    selections['Summary'][state] = {}
                    selections['Summary'][state]['items'] = {}

    
                if key not in selections['Instances'][cnt][state]['items']: selections['Instances'][cnt][state]['items'][key] = label
                if key not in selections['Summary'][state]['items']: selections['Summary'][state]['items'][key] = label

                for cat in ['flower', 'fruit', 'leaf', 'water', 'height', 'width', 'type', 'color']:                 
                    for v in df[df['id'] == int(key)][cat].to_string(index=False).strip('][').replace('\'', '').split(', '):  

                        # Update instance in instances
                        if cat in selections['Instances'][cnt][state].keys():
                            if v in selections['Instances'][cnt][state][cat].keys():
                                selections['Instances'][cnt][state][cat][v] += 1
                            else:
                                selections['Instances'][cnt][state][cat][v] = 1
                        else:
                            selections['Instances'][cnt][state][cat] = {v : 1}   

                        # Update Summary   
                        if cat in selections['Summary'][state].keys():

                            if v in selections['Summary'][state][cat].keys():
                                selections['Summary'][state][cat][v] += 1
                            else:
                                selections['Summary'][state][cat][v] = 1
                        else:
                            selections['Summary'][state][cat] = {v : 1}   

                # Sum Total Aggregation
                


        elif 'btn_reset_survey' in request.form:    
            selections = {}

        else:
            selections['NOFORM'] = 'TRUE'
    #else:
        #selections['GET'] = 'YES'

    write_to_file(selections)

    return render_template('survey.html', samples = samples())

@survey.route("/survey/data")
def json_recommendations():
    return selections

def write_to_file(data):
    
    print("1")
    ip = request.remote_addr
    path = "/var/www/greenthumb/data/sessions/" + ip
    # checking if the directory folder 
    # exist or not.

    
    print("2")
    if not os.path.exists(path):        
        # if the folder directory is not present then create it.
        os.umask(0)
        os.makedirs(path, 0o0777)
    
    
    print("3")
    # presist the session data for filters so we can read between pages and blueprints
    with open(path + '/survey.json', 'w') as f:
        json.dump(data, f)    

    os.umask(0)
    os.chmod(path + '/survey.json', 0o0777)

        
    print("4")


##########################
# Filters for selection
##########################
def attractions(d, f):
    # Attractions
    idxs = []
    for index, row in d.iterrows():
        for i in list(f['SELECTIONS']['attractions'].values()):
            v = row['attracts']
            if i in row['attracts']:      
                idxs.append(index)

    if idxs != []:
        d = d.loc[idxs]

    return d

def targets():
    return ['musk mallow', 'jacob', 'pineapple guava', 'egyptian yarrow',
       'yarrow', 'sneezewort', 'ladybells', 'bishop', 'giant hyssop',
       'hummingbird', 'anise hyssop', 'dwarf hybrid hyssop',
       'purple giant hyssop', 'ornamental onion', 'scallion',
       'german garlic', 'golden garlic', 'garlic chives', 'wild garlic',
       'lemon verbena', 'sweet almond bush', 'shellplant', 'shell ginger',
       'lily of the incas', 'madwort', 'small', 'dill',
       'golden chamomile', 'dyer', 'golden marguerite', 'cape pondweed',
       'columbine', 'sweet', 'mountain rockcress', 'horseradish',
       'wormwood', 'mugwort', 'california sagebrush', 'silky wormwood',
       'white sage', 'western mugwort', 'prairie milkweed',
       'swamp milkweed', 'fourleaf milkweed', 'deadly nightshade',
       'chocolate flower', 'swan river daisy', 'angel', 'brugmansia',
       'maikoa', 'calamint', 'camellia', 'peach', 'caper', 'bush anemone',
       'yellow oleander', 'california lilac', 'red valerian',
       'mexican orange', 'clematis', 'blue jasmine', 'golden', 'clethra',
       'sakaki', 'large', 'colewort', 'sea kale', 'florida swamp lily',
       'crinum', 'daphne', 'carnation', 'pink', 'sweet william',
       'cheddar pink', 'maiden pink', 'dianthus', 'border carnation',
       'garden pinks', 'fringed pink', 'dittany', 'northern dragonhead',
       'daily dew', 'coneflower', 'purple coneflower',
       'pale purple coneflower', 'yellow coneflower',
       'eastern coneflower', 'oleaster', 'foxtail lily',
       'giant desert candle', 'loquat', 'sulphur flower', 'joe pye weed',
       'coastal plain joe pye weed', 'meadowsweet',
       'queen of the prairie', 'dropwort', 'florence fennel',
       'crown imperial', 'fritillary', 'blanket flower', 'gardenia',
       'gaura', 'broom', 'gladiolus', 'peacock gladiolus', 'verbena',
       'glumicalyx', 'false indian plantain', 'white garland',
       'french honeysuckle', 'daylily', 'tetraploid daylily',
       'citron daylily', 'tetrapoild daylily', 'yellow daylily', 'dame',
       'coral bells', 'common hop', 'english bluebell', 'common hyacinth',
       'spider lily', 'round fruited st', 'hyssop', 'spring starflower',
       'tall bearded iris', 'iris', 'tall bearded reblooming iris',
       'dwarf iris', 'reticulated iris', 'southern blue flag', 'itea',
       'red', 'sweet pea', 'english lavender', 'lavandin', 'lavender',
       'cooper', 'summer snowflake', 'trumpet lily', 'longiflorum',
       'orienpet lily', 'lily', 'oriental lily', 'asiatic lily',
       'turkscap lily', 'coral lily', 'sweet alyssum', 'honeysuckle',
       'honeyberry', 'lycoris', 'resurrection lily', 'creeping jenny',
       'magnolia', 'banana magnolia', 'apple', 'dwarf apple',
       'columnar apple', 'standard apple', 'american agave',
       'german chamomile', 'brompton stock', 'lemon mint',
       'bells of ireland', 'bee balm', 'eastern beebalm', 'wild bergamot',
       'dotted beebalm', 'grape hyacinth', 'myrtle', 'double daffodil',
       'poeticus daffodil', 'cyclamineus daffodil', 'trumpet daffodil',
       'split', 'jonquilla daffodil', 'tazetta daffodil',
       'triandrus daffodil', 'species daffodil', 'bulbocodium daffodil',
       'american lotus', 'sacred lotus', 'catmint', 'catnip', 'nepeta',
       'oleander', 'yellow pond lily', 'hardy water lily', 'sweet basil',
       'basil', 'oregano', 'dittany of crete', 'star of bethlehem',
       'giant summer hyacinth', 'fragrant tea olive', 'holly olive',
       'peony', 'fernleaf peony', 'common peony', 'arctic poppy',
       'american feverfew', 'wild quinine', 'blue passionflower',
       'purple passionflower', 'passion flower', 'beardtongue',
       'turkish sage', 'sage', 'garden phlox', 'phlox', 'sand phlox',
       'wild sweet william', 'smooth phlox', 'prairie phlox',
       'ozark phlox', 'skunk plant', 'snakemouth orchid', 'tuberose',
       'milkwort', 'dwarf apricot', 'apricot', 'dwarf plum',
       'sour cherry', 'dwarf peach', 'dwarf nectarine', 'striped squill',
       'puschkinia', 'mountain mint', 'slender mountain mint',
       'common pear', 'dwarf pear', 'pear', 'rhododendron', 'currant',
       'black currant', 'red currant', 'rodgersia',
       'landscape shrub rose', 'sweet coneflower', 'rose gentian',
       'meadow sage', 'clary sage', 'wood sage', 'soapwort',
       'hybrid pitcher plant', 'pitcher plant', 'yellow pitcher plant',
       'yellow trumpet pitcher plant', 'white trumpet pitcher plant',
       'green pitcherplant', 'purple pitcher plant',
       'sweet pitcher plant', 'lizard', 'little bluestem', 'sisymbrium',
       'goldenrod', 'prairie dropseed', 'betony', 'batflower',
       'african marigold', 'french marigold', 'painted daisy',
       'common tansy', 'yellow meadow rue', 'thyme', 'star jasmine',
       'nasturtium', 'society garlic', 'miscellaneous tulip',
       'garden heliotrope', 'fragrant viburnum', 'horned violet', 'pansy',
       'watsonia', 'silky wisteria', 'japanese wisteria',
       'chinese wisteria']


# Randomly fill the survey from current data states and the source
def samples():

    count = 12

    # Get ids with pics
    keys = []
    p = '/var/www/greenthumb/static/assets/images/plants/'
    
    # get only items in the plant master for model
    common = list(plant_df['common'].unique())
    data = df[df['common'].isin(common)]
    # pull only working labels
    data = data[data['common'].isin(targets())]

    for idx in data['id'].tolist():
        f =  str(idx) + '.jpg'
        path = os.path.join(p, f)
        # if path has picture use it
        if os.path.exists(path):
            # if picture is big enough its real
            if os.stat(path).st_size > 10000:
                keys.append(idx)

    random.shuffle(keys)
    samples = {}
    
    # Head
    samples['head'] = {'count': count}
    
    # Body
    samples['body'] = {}
    for i, idx in enumerate(keys[0:count]): #// getting an (i,d) in the html

        samples['body'][i] = {'idx' : str(idx)
                            , 'label': df[df['id'] == idx]['common'].to_list()[0]}

    # random select options to show
    return samples  