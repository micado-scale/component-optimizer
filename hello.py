from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello Istvan'

if __name__ == '__main__':
    # app.run(host='192.168.0.60', debug=True)
    app.run()
 