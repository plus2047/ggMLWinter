import flask
import uuid
import os
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired
from flask import url_for

app = flask.Flask(__name__)
app.secret_key = "this is a secret key."


class InputForm(FlaskForm):
    string = TextAreaField("Input", validators=[DataRequired()])
    submit = SubmitField('submit')


@app.route("/", methods=("GET", "POST"))
def index():
    form = InputForm()
    if form.validate_on_submit():
        pic_list = [url_for("static", filename="Lantingji_Xu/%s.jpg" % i) for i in range(1, 50)]
        return flask.render_template("index.html", form=form, pic_list=pic_list)
    return flask.render_template("index.html", form=form, pic_list=None)
