# Bachelor-Thesis_Ivan-Shtetinski
Open Questions and Essays Assessment Tool

#Pre-requirements:

-Python 3+ and the latest version of pip (https://pip.pypa.io/en/stable/installing/)

#To install the modules that are used, type the following into the terminal:

-pip install pywebview #you might also need to install PyQt(https://pywebview.flowrl.com/guide/installation.html#dependencies)

-pip install pandas

-pip install numpy or pip install -U numpy

-pip install -U scikit-learn==0.21.3

-pip install pyspellchecker

-pip install eli5

-pip install bs4

-pip install wordcloud 
#after installing from scratch on a different machine I came across the following error:Microsoft Visual C++ 14.0 is required:
if you have a similar problem The Solution is: (https://stackoverflow.com/a/49986365/12774828)

    Go to Build Tools for Visual Studio (https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017)

    Select free download under Visual Studio Community 2017. This will download the installer. Run the installer.

    Select what you need under workload tab:

    a. Under Windows, there are 3 choices. Only check Desktop development with C++

    b. Under Web & Cloud, there are 7 choices. Only check Python development (I believe this is optional But I have done it).

-pip install matplotlib

-pip install nltk

to download different packadges from the NLTK module run python via the command prompt(-python) and afterwards tyoe the following:

  import nltk
  
  nltk.download()
  
  A window for the nltk downloader should open up. Go to All Packages and install the ones with the following identifiers: punkt, stopwords, wordnet; Make sure the download directory is accessible.(more info here: https://www.nltk.org/data.html)

#How to run:

-extract the images and data folders in the same directory as the python file.(They aren't really needed, as the program generates new images and you can work with any files on your machine, but for convinience its recomended. In any gase, make sure you have a folder named images in the same directory as the Asessment_Tool.py file)

-run the Asessment_Tool.py file

-upload a training file by clicking the upper-most button and selecting one of the 8 training files from the data folder
-afterwards upload a file, for which you would like to see the visualisations; after it is done a dashboard will open automatically

-both operations might take a bit of time to load

-to view the visualisations for the last uploaded file again, click the last button

-More details on each dataset can be found in data/Essay_Set_Description

Enjoy

