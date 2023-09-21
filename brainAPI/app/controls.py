import os

def delete_files():
    dir = os.listdir("static/uploads/")
    for i in range(0, len(dir)):
        os.remove('static/uploads/'+dir[i])