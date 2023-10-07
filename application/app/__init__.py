from flask import Flask

app=Flask(__name__)

from app import views

# make_static = MakeStatic()
# make_static.init_app(app)
# with app.app_context():
# make_static.compile()