This folder contains the web scraping script which we can use to create our own dataset of images from the web.

We need to install the following dependencies:

1. Create your Cognitive services account on [Bing Image Search API](https://azure.microsoft.com/en-us/try/cognitive-services/my-apis/?api=bing-image-search-api)
2. Click on Get API-key button on the previous webpage to get your personal api key which you can use to communicate with Bing Search API
3. Install the requests package using the following command:
    
    $ pip install requests


   

**WebScraping.py:** is the script we will use to scrape the web for images and create our dataset. Make sure you add your API-Key obtained in step-2 above in the script

Steps to run the script:
1. Make a folder called 'dataset' in the folder which contains WebScraping.py
2. Create a dataset for desired animal (<animal_name_directory>) in the 'dataset' folder created in previous step

Run the script using the following command:

    $ python search_bing_api.py --query "<animal_name>" --output dataset/<animal_name_directory>

 
References:
1. How to (quickly) build a deep learning image dataset tutorial by Adrian Rosebrock which can be referred [here](https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/)