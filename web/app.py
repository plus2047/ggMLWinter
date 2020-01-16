import flask
import uuid
import os
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired
from flask import url_for
import pathlib

app = flask.Flask(__name__)
app.secret_key = "this is a secret key."

DEFAULT_SENTENCE = "大江东去，浪淘尽，千古风流人物。"

char_set = set()

for f in pathlib.Path("static/font").glob("*.png"):
    try:
        char_set.add(int(f.stem))
    except:
        pass

class InputForm(FlaskForm):
    string = TextAreaField("Input", validators=[DataRequired()], default=DEFAULT_SENTENCE)
    submit = SubmitField('submit')


def get_pic_list(s):
    return [url_for("static", filename="font/%s.png" % ord(w)) for w in s if ord(w) in char_set]


@app.route("/", methods=("GET", "POST"))
def index():
    form = InputForm()
    if form.validate_on_submit():
        pic_list = get_pic_list(form.string.data)
        return flask.render_template("index.html", form=form, pic_list=pic_list)
    return flask.render_template("index.html", form=form, pic_list=None)
