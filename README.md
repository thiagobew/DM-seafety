# DM-seafety

Repositório para código utilizado no projeto.

Execute o comando "pip install -r requirements.txt" na root do projeto.

Dataset deve ser baixado separadamente pelo link no drive (https://drive.google.com/file/d/10VBj7siGtwhiWU-7SPgLJvvTFdN3jHb7/view) e extraído para a pasta dataset, em TfCNNClassifiers, trocando o nome da pasta de caddy-gestures-complete-v2-release-all-scenarios-fast.ai para dataset.

Zips da imagem, para serem colocados na pasta ResNet50Classifier/images, pode ser baixado aqui: (https://drive.google.com/drive/folders/1qvLkZpH32jUNhjRIkWGiLUwgu5WLszOd?usp=sharing). E não deve ser extraído, o próprio código faz isso.

Para testar os modelos, basta rodar ResNet50Classifier/ResNetCaddyClassifier.py, TfCNNClassifiers/CNNCaddyClassifier.py e TfCNNClassifiers/LibrasCaddyClassifier.py.
O pre-processamento das imagens é feito automaticamente.

