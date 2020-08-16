from flask import Flask, render_template, Response, send_file
from src import image_counter1
from cv_camera import VideoCamera, audio_processed, audio_changed

path_to_file = "./static/rescources/SOUND.mp3"

app = Flask(__name__)

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def getIndex():
    global image_counter1
    image_counter1 += 1
    return image_counter1

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/haha")
def haha():
    path = './static/rescources/' + str(getIndex()) + '.jpg'
    return render_template("haha.html", currentImage=path)

@app.route("/getImage")
def getImage():
    return render_template("haha.html")

def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# def audio_gen():
#     while True:
#
#         if(audio_changed):
#             path_to_file = "./123.wav"
#
#
#
#         yield (b'--audio\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + audio + b'\r\n\r\n')

@app.route('/sound_feed')
def sound_feed():

    global path_to_file

    while True:
        if(audio_changed):
            path_to_file = "./123.wav"

        return send_file(
            path_to_file,
            mimetype="audio/wav",
            as_attachment=True,
            attachment_filename=path_to_file)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003, debug=True)