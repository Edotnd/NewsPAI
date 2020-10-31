from flask import Flask, render_template
import pandas as pd
# app = Flask(__name__)

news_data = pd.read_table("upload_file.txt", error_bad_lines=False, sep='\t', names=['label', 'url', 'sentence'])
news_data.dropna(axis=0)

for i, j in news_data.groupby('label'):
    print(j)

'''@app.route('/')
def index():
    return render_template('index.html', data=news_data)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8080")'''