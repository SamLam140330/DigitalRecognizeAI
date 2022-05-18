import os

import turicreate as tc

tc.config.set_num_gpus(1)


def load_model_from_path():
    data = tc.image_analysis.load_images('digit_images', with_path=True)
    data['label'] = data['path'].apply(lambda
                                           path: '0' if '/zero' in path else '1' if '/one' in path else '2' if '/two' in path else '3' if '/three' in path else '4' if '/four' in path else '5' if '/five' in path else '6' if '/six' in path else '7' if '/seven' in path else '8' if '/eight' in path else '9')
    data.save('digits.sframe')


def train_model():
    data = tc.SFrame('digits.sframe')
    train_data, test_data = data.random_split(0.8)
    model = tc.image_classifier.create(train_data, target='label', max_iterations=500)
    metrics = model.evaluate(test_data)
    print(metrics['accuracy'])
    model.save('digits.model')
    model.export_coreml('my_custom_image_classifier.mlmodel')


def predict_model(model_name):
    model = tc.load_model('digits.model')
    image_data = tc.image_analysis.load_images(model_name)
    predictions = model.predict(image_data, output_type='class')
    print("The prediction is " + predictions)


load_model_from_path()
train_model()
predict_model(os.path.abspath(os.getcwd()) + "/uploads/00017.png")
