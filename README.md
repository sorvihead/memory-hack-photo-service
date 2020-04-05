#Readme
##What is it
Сервис позволяющий оцифровать фотографию или определить степень похожести двух фото по Rest API
## Requirements
* flask
* opencv==3.4.0
* tensorflow==1.15.0
* keras
* Pillow
* h5py
* sckikit-image
* scipy
* numpy

##How to build 
* Необходим дистрибутив Conda
* Устанавливаем зависимости через conda install
* Загружаем в facenet.model https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view
* Загружаем в align_mtcnn_service.mtcnn_model https://github.com/wangbm/MTCNN-Tensorflow/tree/master/save_model
* Загружаем набор цветных фотографий в digitalization_photo_service.data.testdata.Train и в Validate ЧБ
 фото для обучения модели. Либо можно воспользоваться уже обученными моделями, но тогда нужно загрузить набор ЧБ фото в 
 Validate, тк модель проходит валидацию перед стартом.
* Удаляем плейсхолдеры из папок где должны лежать данные (data, people)
* flask run
