from flask import Blueprint, render_template, jsonify
import pandas as pd

cart = Blueprint('cart', __name__)
#df = pd.read_csv('./data/plants.csv')     


##########################
# Flask routes
##########################
@cart.route("/cart")
def ui():   
    return render_template('cart.html')

def add_to_cart():
    pass

def remove_from_cart():    
    pass