Jacob Szczudlik - part 1 - CV engineer - Modeling for dataset analysis

--Running ImgClassifier.html

ImgClassifier.html is the link to the working fruit image classification website. here you can upload an image of the fruit you want classified and the model will return to you the predicition for the fruit and its variation. App.html is a static html and uses the file:// protocol (not http://). So you should only need to run the backend and not a server. but if for some reason that doesnt work ill include intructions to fire up a python server to serve html. 


1. **Install dependencies**:

$python -m venv venv
$source venv/bin/activate  # On Windows: venv\Scripts\activate

$pip install -r requirements.txt


2. **run flask backend**:

$python backend.py

 you should see something like  -> Running on http://127.0.0.1:5000/

3. open ImgClassifier.html
   
just double click ImgClassifier.html in the directory.






Hanwen Dong - part2 - NLP Engineer: Semantic Model for Recipe Retrieval
