from flask import Flask, render_template, Response, redirect, url_for
from camera_yolo import CameraYOLO

app = Flask(__name__)
camera = CameraYOLO(model_path="newest3.pt", line_position=300)

@app.route("/")
def index():
    return render_template("index.html", count=camera.total_count)

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(gen(camera), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/reset", methods=["POST"])
def reset():
    camera.total_count = 0
    return redirect(url_for("index"))

@app.route("/count")
def count():
    """Return live count (AJAX polling)"""
    return {"count": camera.total_count}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
