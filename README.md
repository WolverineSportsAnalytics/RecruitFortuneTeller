### Instillation Instructions

#### Clone the repository 
Make a directory 

`git clone https://github.com/WolverineSportsAnalytics/RecruitFortuneTeller.git`

#### Install Virtual Env 
Virtual Env acts as a virtual enviornment so that we can virtually install python packages and not overwrite the ones 
on our system 

`pip install virtualenv`

#### Create Virtual Env and Enter it 
Go to your directory where you cloned the repository 

`$ virtualenv ENV`

`$ source bin/activate`

#### Install the requirements
 
`pip install ../requirements.txt`

#### Deactivate the Virtual ENV
`$ deactivate`

##### Extra: If You Want to Install A New Package
`$ source bin/activate`

`pip install package`

`pip freeze > ../requirements.txt`

Commit the requirements.txt file 

### How to Run 

All test.py does right now is that it gets an access code via OAuth2 protocol and makes a request to get a timelines 
tweet. Run in pycharm to see what the data object returns 

`python test.py`