from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, IntegerField
from wtforms.validators import DataRequired, Length, Email, EqualTo, NumberRange


class RegistrationForm(FlaskForm):
    # username = StringField('Username',
    #                        validators=[DataRequired(), Length(min=1, max=10)])
    age = IntegerField('Age',validators=[DataRequired(),NumberRange(min=1, max=150)])
    bmi = IntegerField('Bmi',validators=[DataRequired(),NumberRange(min=5, max=50)])
    # email = StringField('Email',
    #                     validators=[DataRequired(), Email()])
    # password = PasswordField('Password', validators=[DataRequired()])
    # confirm_password = PasswordField('Confirm Password',
    #                                  validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Submit')


# class LoginForm(FlaskForm):
#     email = StringField('Email',
#                         validators=[DataRequired(), Email()])
#     password = PasswordField('Password', validators=[DataRequired()])
#     remember = BooleanField('Remember Me')
#     submit = SubmitField('Login')
