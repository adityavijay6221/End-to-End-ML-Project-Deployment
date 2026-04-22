from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@application.route('/')
def index():
    return render_template('index.html')


@application.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')

    data = CustomData(
        gender=request.form.get('gender'),
        race_ethnicity=request.form.get('race_ethnicity'),
        parental_level_of_education=request.form.get('parental_level_of_education'),
        lunch=request.form.get('lunch'),
        test_preparation_course=request.form.get('test_preparation_course'),
        math_score=request.form.get('math_score'),
        reading_score=request.form.get('reading_score'),
        writing_score=request.form.get('writing_score'),
    )

    pipeline = PredictPipeline()
    result = pipeline.predict(data.as_dataframe())
    return render_template('home.html', result=round(result[0], 2))


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8080, debug=False)
