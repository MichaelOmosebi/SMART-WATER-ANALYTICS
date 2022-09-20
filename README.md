# A PROJECT ON WATER AVAILABILITY MANAGEMENT


## 1) Summary of the Project
The Acea Group is one of the leading Italian multiutility operators. Listed on the Italian Stock Exchange since 1999, the company manages and develops water and electricity networks and environmental services. Acea is the foremost Italian operator in the water services sector supplying 9 million inhabitants in Lazio, Tuscany, Umbria, Molise, Campania.

This project focuses on the water sector to help Acea Group preserve precious waterbodies. As it is easy to imagine, a water supply company struggles with the need to forecast the water level in a waterbody (water spring, lake, river, or aquifer) to handle daily consumption. During fall and winter waterbodies are refilled, but during spring and summer they start to drain. To help preserve the health of these waterbodies it is important to predict the most efficient water availability, in terms of level and water flow for each day of the year.

The desired outcome is a notebook that can generate four mathematical models, one for each category of waterbody (```acquifers```, ```water springs```, ```river```, ```lake```) that might be applicable to each single waterbody.



Kaggle Link: https://www.kaggle.com/competitions/acea-water-prediction

For this repository, please refer to the following files:

| File Name              | Description                                 |
| :--------------------- | :--------------------                       |
| `ACEA_App.py`          | Streamlit application definition.           |
| `Ausers_Aquifer.ipynb` | Auser dataset code file - Intro to Project  |
| `Amiata Spring.ipynb`  | Spring dataset code file                    |
| `River Arno.ipynb`     | River dataset code file                     |
| `Lake Bilancino.ipynb` | Lake dataset code file                      |


## 2) Project Usage Instruction

#### 2.1) Creating a copy of this repo

||| Kindly ```fork``` the repo and upvote if you find helpful.

![Fork Repo](resources/imgs/fork-repo.png)  

To fork the repo, simply ensure that you are logged into your GitHub account, and then click on the 'fork' button at the top of this page as indicated within the figure above.

#### 2.2) Running the Streamlit web app on your local machine

As a first step to becoming familiar with our web app's functioning, we recommend setting up a running instance on your own local machine.

To do this, follow the steps below by running the given commands within a Git bash (Windows), or terminal (Mac/Linux):

 1. Ensure that you have the prerequisite Python libraries installed on your local machine:

 ```bash
 pip install -U streamlit numpy pandas scikit-learn
 ```

 2. Clone the *forked* repo to your local machine.

 ```bash
 git clone https://github.com/{your-account-name}/classification-predict-streamlit-template.git
 ```  

 3. Navigate to the base of the cloned repo, and start the Streamlit app.

 ```bash
 streamlit run ACEA_App.py
 ```

 If the web server was able to initialise successfully, the following(or something similar) message should be displayed within your bash/terminal session:

```
  You can now view your Streamlit app in your browser.

    Local URL: http://localhost:8501
    Network URL: http://192.168.43.41:8501
```

You should also be automatically directed to the base page of your web app. This should look something like:

![Streamlit base page](resources/imgs/streamlit-base-splash-screen.png)

Thanks for interacting! Recommendations are welcome.