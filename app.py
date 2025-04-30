from flask import Flask, render_template, request
import model

app = Flask(__name__)

# List of valid user IDs
VALID_USER_IDS = [
    '00sab00', '1234', 'zippy', 'zburt5', 'joshua',
    'dorothy w', 'rebecca', 'walker557', 'samantha',
    'raeanne', 'kimmie', 'cassie', 'moore222'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_name = request.form.get('User Name')

    if user_name in VALID_USER_IDS:
        top20 = model.recommendtop20_products(user_name)
        top5 = model.recommendtop5_products(top20)
        return render_template(
            'index.html',
            column_names=top5.columns,
            row_data=top5.values.tolist(),
            zip=zip,
            text='Recommended products'
        )
    else:
        return render_template('index.html', text='No recommendation found for this user')

if __name__ == '__main__':
    app.run(debug=False)
