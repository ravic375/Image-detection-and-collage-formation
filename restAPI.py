from flask import Flask, request, jsonify, send_file
from flask_restful import Resource, Api
import project_main
import os
import cv2
app = Flask(__name__)
api = Api(app)

class Index(Resource):
    def get(self):
        json_output = {
            "Message" : "Ciphense Internship Challenge",
        }
        return json_output


class ImageDetails(Resource):
    def evalClasses(self, classes):
        animals = ['bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe']

        json_output = {
            "Message": "NULL",
            "Data": [
                {
                    "PersonCount": 0
                },
                {
                    "AnimalCount": 0,
                    "Animals": []
                },
                {
                    "ObjectCount": 0
                }
            ]
        }
        json_data = json_output["Data"]
        for class_name in classes:
            if class_name in animals:
                json_data[1]["AnimalCount"] += 1
                json_data[1]["Animals"].append(class_name)
            elif class_name == 'person':
                json_data[0]["PersonCount"] += 1
            else:
                json_data[2]["ObjectCount"] += 1
        return json_output
        
    def post(self):
        file = request.files['image']
        file_path = os.path.join('./imgs/', file.filename)
        file.save(file_path)
        classes , _ = project_main.objectDetector(imagePath=file_path)
        json_output = {}
        json_output = self.evalClasses(classes)
        if json_output:
            json_output["Message"] = "Image Details Extracted Successfully"
        return json_output


class Collage(Resource):
    def post(self):
        imgId = 0
        files = request.files.getlist("image")
        if len(files) > 0:
            os.system('del /q .\collage_pics\*')
        for file in files:
            file_path = os.path.join('./imgs/', file.filename)
            file.save(file_path)
            image, detected_faces = project_main.faceDetector(file_path)
            for (col, row, width, height) in detected_faces:
                file_name = 'img' + str(imgId) + '.jpg'
                imgId += 1
                cropped_file_path = os.path.join('./collage_pics/', file_name)
                cropped_image = image[row-10: row + width + 10, col-10: col+height+10]
                cv2.imwrite(cropped_file_path, cropped_image)
        image = project_main.createCollage()
        cv2.imwrite('./collage_pics/collage.jpg', image)
        return send_file('./collage_pics/collage.jpg', mimetype='image/jpg')


api.add_resource(Index, '/')
api.add_resource(ImageDetails,'/getImageDetails')
api.add_resource(Collage,'/createCollage')


if __name__ == '__main__':
    app.run(host='0.0.0.0')

