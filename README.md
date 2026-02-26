az configure --defaults group=MS workspace=iris_demo

<- Specify which workspace in azure ml you want to work on


az ml environment create -f environment/environment.yml

<- Creating an Environment


az ml job create -f job/job.yml

<- Creating an Job File


az ml model create -f model/model.yml

<- Creating an Model File


mkdir -p demo_model

<- Create a folder for downloading Model files


az ml model download \
  -n iris-model \
  -v 1 \
  --download-path ./demo_model/iris-model
  
<- Download the model asset registered in Azure ML, specify the model name and model version, and write which path to download.

 
python src/score.py --model_dir ../demo_model/iris-model/model
<- Import model files from the path of ../demo_model/iris-model/model and classify them based on the predicted values set by src/score.py

