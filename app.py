import os
from flask import Flask, request, flash, render_template, jsonify, json,  redirect, url_for, session
from function import preprocess_data, result_svm
import csv
import pandas

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

app = Flask(__name__)

users = {
    'admin': ('admin', 'admin'),
    'user1': ('password1', 'user'),  # Contoh pengguna biasa
    'user2': ('password2', 'user')  # Tambah lebih banyak jika diperlukan
}
app.secret_key = 'supersecretkey'



# Rute untuk halaman utama
@app.route('/')
def home():
    if 'username' in session:
        role = session.get('role')
        if role == 'admin':
            return redirect(url_for('dashboard'))
        elif role == 'user':
            return redirect(url_for('site'))
    return render_template('home.html')


#upload files
app.config['UPLOAD_FOLDER']='uploads'
ALLOWED_EXTENSION = set(['csv'])

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION
  
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username][0] == password:
            session['username'] = username
            session['role'] = users[username][1]  # Simpan role di sesi
            if users[username][1] == 'admin':
                return redirect(url_for('dashboard'))
            elif users[username][1] == 'user':
                return redirect(url_for('site'))
        else:
            return 'Invalid credentials, please try again!'
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'username' in session and session.get('role') == 'admin':
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/site')
def site():
    if 'username' in session and session.get('role') == 'user':
        return render_template('site.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/upload-file', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'GET':
    return render_template('uploaddata.html')
    
  elif request.method == 'POST':
    # check if the post request has the file part
    if 'file' not in request.files:
      return redirect(request.url)

    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
      return redirect(request.url)

    if file and allowed_file(file.filename):
      file.filename = "dataset.csv"
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
      text = pandas.read_csv('uploads/dataset.csv', encoding='latin-1')
      # result = text.to_json(r'uploads/dataset.json')
      return render_template('uploaddata.html',tables=[text.to_html(classes='table table-bordered', table_id='dataTable')])

@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocess():
  return render_template ('preprocessing.html')


#@app.route('/preprocessing/result', methods=['GET', 'POST'])
#def preprocessing():
#  text = pandas.read_csv('uploads/dataset.csv', encoding='latin-1')
#  text.drop(['Date','Author'], axis=1, inplace=True)
#  text['Text'] = text['Text'].apply(lambda x:preprocess_data(x))
#  text.to_csv('uploads/dataset_clear.csv', index = False, header = True)
#  return render_template('preprocessing.html',tables=[text.to_html(classes='table table-bordered', table_id='dataTable')])
import pandas as pd  # Pastikan pandas diimpor

@app.route('/preprocessing/result', methods=['GET', 'POST'])
def preprocessing():
    # Membaca dataset dengan encoding latin-1
    text = pd.read_csv('uploads/dataset.csv', encoding='latin-1')
    
    # Menghapus kolom 'Date' dan 'Author'
    text.drop(['english', 'polarity'], axis=1, inplace=True)
    
    # Pastikan semua data dalam kolom 'Text' adalah string
    text['text_1'] = text['text_1'].fillna('').astype(str)  # Ganti NaN dengan string kosong dan konversi ke string
    
    # Terapkan fungsi preprocess_data pada kolom 'Text'
    text['text_1'] = text['text_1'].apply(lambda x: preprocess_data(x))
    
    # Simpan hasil ke file baru
    text.to_csv('uploads/dataset_clear.csv', index=False, header=True)
    
    # Render tabel hasil ke template HTML
    return render_template('preprocessing.html', tables=[text.to_html(classes='table table-bordered', table_id='dataTable')])

# def data(text):
#     text['label'] = text['label'].map({'positif': 2, 'negatif': 1, 'netral': 0})
#     X = text['Text'].fillna(' ')
#     y = text['label']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2)
#     return X_train, X_test, y_train, y_test

score = [
  'POSITIF', 'NETRAL', 'NEGATIF'
]

colors = [
  '#1cc88a', '#e74a3b', '#f6c23e'
]

@app.route('/grafik-data', methods=['GET', 'POST'])
def page():
  return render_template ('klasifikasisvm.html')


@app.route('/grafik-data/result', methods=['GET', 'POST'])
def klasifikasisvm():
  text = pandas.read_csv('uploads/dataset_clear.csv', encoding='latin-1')

  accuracy_rbf, y_test = result_svm(text)
  accuracy_rbf = (round(accuracy_rbf, 2) * 100)
  
  y_test = y_test.reset_index()
  netral, negatif, positif = y_test['score'].value_counts()
  total = positif + negatif + netral
  # print(y_test['label'].value_counts() )

  pie_score = score
  pie_colors = colors
  pie_values = [positif, negatif, netral]

  bar_score = score
  bar_values = [positif, negatif, netral]
  
  return render_template ('klasifikasisvm.html', total_dataset = 500, sentimen_positive = positif, sentimen_negative = negatif, sentimen_netral = netral, total_tweet = total, accuracy_rbf = accuracy_rbf, labels = pie_score, colors = pie_colors, values = pie_values, bar_labels = bar_score, bar_values = bar_values)

@app.route('/tesmodel', methods=['GET', 'POST'])
def tesmodelpage():
  return render_template ('tesmodel.html')

@app.route('/tesmodel/result', methods=['GET', 'POST'])
def tesmodel():
  # Loading model to compare the results
  model = pickle.load(open('uploads/rbf.model','rb'))
  vectorizer = pickle.load(open('uploads/vectorizer.model','rb'))

  text = request.form['text']
  original_text = request.form['text']

  hasilprepro = preprocess_data(text)
  hasiltfidf = vectorizer.transform([hasilprepro])

  # cek prediksi dari kalimat
  
  hasilsvm = model.predict(hasiltfidf)
  if hasilsvm >= 1:
    hasilsvm = 'POSITIF'
  elif hasilsvm == 0:
    hasilsvm = 'NETRAL'
  else:
    hasilsvm = 'NEGATIF'
  
  return render_template ('tesmodel.html', original_text=original_text, hasilprepro=hasilprepro, hasilsvm=hasilsvm)

if __name__ == "__main__":
  app.run(debug=True)