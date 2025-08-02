Jacob Szczudlik - part 1 - CV engineer - Modeling for dataset analysis

--Running ImgClassifier.html

ImgClassifier.html is the link to the working fruit image classification website. here you can upload an image of the fruit you want classified and the model will return to you the predicition for the fruit and its variation. App.html is a static html and uses the file:// protocol (not http://). So you should only need to run the backend and not a server. but if for some reason that doesnt work ill include intructions to fire up a python server to serve html. 


1. **Install dependencies**:

$python -m venv venv
$source venv/bin/activate  # On Windows: venv\Scripts\activate  

the above steps will create a python virtual env to avoid dependecy issues if needed.



$pip install -r requirements.txt


2. **run flask backend**:

$python backend.py

 you should see something like  -> Running on http://127.0.0.1:5000/

3. open ImgClassifier.html
   
just double click ImgClassifier.html in the directory




1. exiting virtual envirnment
   
   $deactivate





optional: running python server

-split terminal

-in first terminal have backend.py running

- in second terminal run $python -m http.server 8000

in your browser copy paste http://localhost:8000/ImgClassifier.html



SAMPLE IMAGES TO TEST ON:
there is a folder in the directory named sampleImgs. This folder contains a few images that you can use to test the classifier if you dont have your own handy! 




Hanwen Dong- Part 2 - NLP Engineer








