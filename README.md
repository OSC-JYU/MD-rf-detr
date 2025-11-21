
# MD-rf_detr

## Introduction

MD-rf_detr allows running rfdetr based models. This service was build to run segmentation model of the Finnisn National Archive.

The codebase is from https://github.com/Kansallisarkisto-AI/rfdetr_trocr_pipeline

model: https://huggingface.co/Kansallisarkisto/rfdetr_textline_textregion_detection_model


### Functionality

The API offers three main services:

1. **Detect lines** - Create line polygons.




## API

endpoint is http://localhost:9011/process

Payload is queue message as json file. 

## Running as service (locally)


Then build and start

	make build
	make start

or start container directly

 	docker run --name md-rf_detr -p 9011:9011  



### Example API call 

Run from MD-RF_DETR directory:

Detect line polygons:

	curl -X POST -H "Content-Type: multipart/form-data" \
	  -F "message=@test/detect.json;type=application/json" \
	  -F "content=@test/htr3.jpg;type=image/jpeg" \
	  http://localhost:9011/process


Similarity index creation (httpie version):

	http POST :9009/process message@test/similarity.json content@test/text_fi.txt --form



