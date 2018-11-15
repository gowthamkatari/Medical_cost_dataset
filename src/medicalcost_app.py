import pickle
import numpy as np
from flask import Flask,render_template, request, url_for, flash, redirect
from sklearn.preprocessing import StandardScaler
from forms import RegistrationForm


sc = StandardScaler()

model_file = 'model/model_binary.sav'
standard_scalar_file = 'model/scalar.sav'
my_model = pickle.load(open(model_file,'rb'))
sc = pickle.load(open(standard_scalar_file,'rb'))

app = Flask(__name__)
app.config['SECRET_KEY'] = '502e21598a751819a7503fb3a2e25911'

@app.route("/")
@app.route("/home", methods=['GET', 'POST'])
def home():
    form = RegistrationForm()
    if form.validate_on_submit():
    	x = use_model(form.age.data,form.bmi.data)
    	return render_template('home_nr.html', data = x)
    return render_template('home.html', title='home', form=form)


# @app.route("/home_nr")
# def home_nr():
# 	x = use_model(25,100)
# 	return render_template('home_nr.html', data = x)

def use_model(age, bmi_value):
	c= [[age,bmi_value]]
	c = sc.transform(c)
	charge_value = my_model.coef_[0]*(c[0][0]) + my_model.coef_[1]*(c[0][1]) + my_model.intercept_
	charge_value = np.exp(charge_value)
	x = ('The Insurance Charges for a {:.1f} years old person who is a Smoker with an bmi = {:.1f} will be {:.4f}'.format(age,bmi_value,charge_value))
	return x
	
@app.route("/about")
def about():
    return render_template('about.html', title='About')




if __name__ == '__main__':
	app.run(port = 9000, debug = True)
