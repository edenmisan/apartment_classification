
from flask import Flask, render_template, request
import pandas as pd
import pickle
from assets_data_prep import prepare_data

app = Flask(__name__)

# טעינת המודל
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# טעינת הדאטה לאימון כדי לייצר את הגלובלים
df_train = pd.read_csv("train.csv")
_ = prepare_data(df_train, mode="train")

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {key: request.form.get(key) for key in request.form.keys()}

        numeric_fields = ["room_num", "floor", "area", "garden_area", "days_to_enter", "num_of_payments",
                          "monthly_arnona", "building_tax", "total_floors", "has_parking", "has_storage",
                          "elevator", "ac", "handicap", "has_bars", "has_safe_room", "has_balcony",
                          "is_furnished", "is_renovated", "distance_from_center"]
        for field in numeric_fields:
            input_data[field] = float(input_data.get(field, 0))

        df_input = pd.DataFrame([input_data])
        processed_input = prepare_data(df_input, mode="test")
        prediction = model.predict(processed_input)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return f"שגיאה: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=81)
