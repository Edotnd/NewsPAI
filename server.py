from flask_ngrok import run_with_ngrok
from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)
run_with_ngrok(app)

shit_list = open('upload_file.txt', 'r')

t_data = []
f_data = []

for line in shit_list.readlines():
  dot = line.split('\t')
  if int(dot[0]) == 1:
    t_data.append([dot[1], dot[2]])
  elif int(dot[0]) == 0:
    f_data.append([dot[1], dot[2]])

@app.route('/')
def index():
    return render_template('index.html', t_data=t_data, f_data=f_data)

if __name__ == '__main__':
    app.run()
