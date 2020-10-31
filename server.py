from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

news_data = pd.read_table("upload_file.txt", error_bad_lines=False, sep='\t', names=['label', 'url', 'sentence'])

t_data=[]
f_data=[]

for i in range(0, news_data.shape[0]):
    try:
        n = news_data.loc[i]
        if int(n[0]) == 1:
            t_data.append([n[1], n[2]])
        elif int(n[0]) == 0:
            f_data.append([n[1], n[2]])
    except:
        news_data.drop([i])

@app.route('/')
def index():
    return render_template('index.html', t_data=t_data, f_data=f_data)

@app.route('/true')
def true():
    return render_template('true.html', data=t_data)

@app.route('/false')
def false():
    return render_template('false.html', data=f_data)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8080")