
# coding: utf-8

# In[3]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[4]:


app = Flask(__name__)
model=pickle.load(open('model_rfr.pkl','rb'))


# In[4]:


@app.route('/')
def home():
    return render_template('index.html')


# In[ ]:


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #It will take inputs from html from all the forms in tex fields and store it in int_features
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    
    #After getting output render index.html but i will give some data which is prediction_text which gets replaced here
    return render_template('index.html', prediction_text='Departure delay is {} min'.format(output))


# In[ ]:


if __name__ == "__main__":
    app.run(debug=True)

