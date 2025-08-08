import gradio as gr
import pickle
import numpy as np

scaler =pickle.load(open('scaler.pkl','rb'))
regressor = pickle.load(open('regressor.pkl','rb'))



def calculategoldrate(usd_inr):
  scaled_input = scaler.transform(np.array(usd_inr).reshape(1,-1))
  return regressor.predict(scaled_input)[0][0].round(2)

sample= gr.Interface(
    fn= calculategoldrate,
    inputs = ["number"],
    outputs =["number"],
    title ='How much is gold rate now?'
)
sample.launch()