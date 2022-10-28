# Project Documentation
Rock-Paper-Scissors is a game in which each player simultaneously shows one of three hand signals representing rock, paper, or scissors. Rock beats scissors. Scissors beats paper. Paper beats rock. The player who shows the first option that beats the other player's option wins. This Project is an implementation of an interactive Rock-Paper-Scissors game, in which the user can play with the computer using the camera.

## Milestone 1:
Set up dev environment

## Milestone 2: Create the computer vision system
### Task 1:
Create an image project model with four different classes: Rock, Paper, Scissors, Nothing:-

Using Google Teachable Machine (https://teachablemachine.withgoogle.com/) create a model where each class is trained with images of yourself showing each option to the camera. The "Nothing" class represents the lack of option in the image.

### Task 2:

Download the model from the "Tensorflow" tab in Teachable-Machine. The model should be named keras_model.h5 and the text file containing the labels should be named labels.txt.
The downloaded files contain the structure and the parameters of a deep learning model. These files are not executable and readable. Add the downloaded files in the Python project.

## Milestone 3: Install the dependencies
### Task 1:
Create a new virtual environment - for Mac users:-
After installing miniconda, create a virtual environment by running the commands below:-

commands:
>conda create -n tensorflow-env python=3.9

>conda activate tensorflow-env

>conda install pip

Then, follow the steps from the section that says "arm64: Apple Silicon" from this link
>https://developer.apple.com/metal/tensorflow-plugin/

After getting tensorflow for Mac, install opencv for Mac by running the following commands:
>conda install -c conda-forge opencv

### Task 2:
Complete the installation of dependencies:-
Install ipykernel by running the following command:
>pip install ipykernel

Once  installed (regardless of the operating system), create a requirements.txt file by running the following command:
>pip list > requirements.txt

This file will contain all dependencies and its version used in the environment created above. 

### Task 3:
Check if the model works:-

```python
import cv2
from keras.models import load_model
import numpy as np
model = load_model('keras_model.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True: 
    ret, frame = cap.read()
    resized_frame2 = cv2.resize(frame, (960, 720), interpolation = cv2.INTER_AREA)
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    prediction = model.predict(data)
    cv2.imshow('frame', resized_frame2)
    # Press q to close the window
    print(prediction)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(int(cap.get(3)))
print(int(cap.get(4)))         

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
```

### Task 4:
Tensorflow and Keras, both libraries are used to build deep learning models (neural networks)
The variable predictions contains the output of the model, and each element in the output corresponds to the probability of the input image representing a particular class.
So, for example, if the prediction has the following output: [[0.8, 0.1, 0.05, 0.05]], there is an 80% chance that the input image shows rock, a 10% chance that it shows paper, a 5% chance that it shows scissors, and a 5% chance that it shows nothing.

The prediction is a numpy array with one row and four columns. So first, we need to access the first row, and then get the index of the highest value in the row.

## Milestone 4: Create Rock, Paper, Scissors game
### Task 1: 
Store the users and computer choice:-
This code needs to randomly choose an option (rock, paper, or scissors) and then ask the user for an input.
Create another file called manual_rps.py that will be used to play the game without the camera. 
Using random module function, pick a random option between rock, paper, and scissors and the input function to get the user's choice.

Create two functions: 'get_computer_choice' and 'get_user_choice'. The first function will randomly pick an option between "Rock", "Paper", and "Scissors" and return the choice. The second function will ask the user for an input and return it.

```python
import random

def get_computer_choice():
    '''
    This function will randomly pick an option between "Rock", "Paper", and "Scissors" 
    and return the choice.
    '''
    game = ["Rock", "Paper", "Scissors"]
    return random.choice(game)
def get_user_choice():
    '''
    This function will ask the user for an input and return it. 
    '''
    user_input = input("Enter your choice - Rock, Paper or Scissors: ")
    return user_input
```
### Task 2:
Figure out who won:-
Using if-elif-else statements, the script should now choose a winner based on the classic rules of Rock-Paper-Scissors.
For example, if the computer chooses rock and the user chooses scissors, the computer wins.
Will wrap the code in a function called get_winner and return the winner. This function takes two arguments: computer_choice and user_choice.

```python
def get_winner(computer_choice, user_choice):
    '''
    This function will choose a winner based on the classic rules of Rock-Paper-Scissors.
    This function takes two arguments: computer_choice and user_choice.
    '''
    computer_choice = computer_choice.lower()
    user_choice = user_choice.lower()
    
    if computer_choice == user_choice:
        print(f"Game draw! Computer's choice is '{computer_choice}'. Your choice is '{user_choice}'")
    elif computer_choice == "rock" and user_choice == "paper":
        print(f"You win! Computer's choice is '{computer_choice}'. Your choice is '{user_choice}'")   
    elif computer_choice == "paper" and user_choice == "rock":
        print(f"You lost! Computer's choice is '{computer_choice}'. Your choice is '{user_choice}'")
    elif computer_choice == "rock" and user_choice == "scissors":
        print(f"You lost! Computer's choice is '{computer_choice}'. Your choice is '{user_choice}'")   
    elif computer_choice == "scissors" and user_choice == "rock":
        print(f"You win! Computer's choice is '{computer_choice}'. Your choice is '{user_choice}'")
    elif computer_choice == "paper" and user_choice == "scissors":
        print(f"You win! Computer's choice is '{computer_choice}'. Your choice is '{user_choice}'")   
    elif computer_choice == "scissors" and user_choice == "paper":
        print(f"You lost! Computer's choice is '{computer_choice}'. Your choice is '{user_choice}'")
```

### Task 3:
Create a function to simulate the game:-
All the code programmed so far relates to one thing: running the game - so we should wrap it all in one function.
Create and call a new function called 'play'.
Inside this function all three functions are called (get_computer_choice, get_user_choice, and get_winner).
Now after runing the code, it should play a game of Rock-Paper-Scissors, and it should print whether the computer or user won.

```python
def play():
    '''
    Inside this function, following four functions are called : 
    get_computer_choice,
    countdown, 
    get_user_choice, and
    get_winner
    When this function is called, User play a game of Rock-Paper-Scissors, 
    and it prints whether the Computer or User has won.
    '''
    game_choice = ["Rock", "Paper", "Scissors", "Nothing"]
    computer_choice = game_choice[0:3]
    user_choice = game_choice
    game = rps(computer_choice,user_choice)

    while True:
        while True:
            c_choice = game.get_computer_choice()
            game.countdown()
            u_choice = game.get_prediction()
            game.get_winner(c_choice, u_choice)
            print(f"Computer wins - {game.computer_wins} & User wins - {game.user_wins}" )
            if game.computer_wins == 3 or game.user_wins == 3:
                break
            
        replay = input("Do you want to play again (y/n)?")
        if replay == 'y':
            game = rps(computer_choice,user_choice)
        else:
            print("Good Bye!")
            break   
          
    # After the loop release the cap object
    game.cap.release()   
    # Destroy all the windows
    cv2.destroyAllWindows()   

play()
```

## Milestone 5: Use the camera to play Rock, Paper, Scissors.
### Task 1:

Create a new file called camera_rps.py where trained prediction model is used to access user input through webcam/camera.
Create a new function called 'get_prediction' that will return the output of the model used earlier.
The output of the model downloaded is a list of probabilities for each class. Code should be defined to pick the class with the highest probability. So, for example, assuming to trained the model in this order: "Rock", "Paper", "Scissors", and "Nothing", if the first element of the list is 0.8, the second element is 0.1, the third element is 0.05, and the fourth element is 0.05, then, the model predicts that you showed "Rock" to the camera with a confidence of 0.8.

The model can make many predictions at once if given many images. In case of giving one image at a time. That means that the first element in the list returned from the model is a list of probabilities for the four different classes.

```python
def get_prediction(self):
        '''
        This function return the output of the model trained for RPS game.
        '''
        start = time.time()
        while (time.time()-start) < 5: 
            ret, frame = self.cap.read()
            resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            #resized_frame2 = cv2.resize(frame, (960, 720), interpolation = cv2.INTER_AREA)
            image_np = np.array(resized_frame)
            normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
            self.data[0] = normalized_image
            prediction = self.model.predict(self.data)
            font = cv2.FONT_HERSHEY_SIMPLEX
            resized_frame = cv2.putText(resized_frame, f"Show your choice in 5 secs.", (0, 20), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
            resized_frame = cv2.putText(resized_frame, f"Countdown :- {5-int(time.time() - start)} secs", (0, 30), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow('frame', resized_frame)
            # Press q to close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return prediction
```
### Task 2
Count down:
In the previous task, the script reads the input from the camera and then compares it with the computer's choice without stopping. However, while playing a regular game, we usually count down to zero, and at that point show hand.
Add a countdown using the function time.time() to get how much time has passed since the script started.

```python
def countdown(self):
        '''
        The function time.time() is used to get how much time has passed since the script started.
        The countdown function gives the user 3 sec to show the choice in front of the camera.
        '''
        print("Game is about to start. Show your choice in front of the camera (Rock, Paper, Scissors)")
        #the time you want
        wait_time = 3
        start_time = time.time()
        while (time.time() - start_time) < wait_time:
            pass
```
### Task 3:
Repeat until a player gets three victories:
The game should be repeated until either the computer or the user wins three rounds.

### Task 4:
Create a class, this class defines the initialization parameters and all the methods required to play RPS game.


```python
class rps():
    def __init__(self,computer_choice, user_choice):
        self.computer_choice = computer_choice
        self.user_choice = user_choice
        self.model = load_model('keras_model.h5')
        self.cap = cv2.VideoCapture(0)
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.computer_wins = 0
        self.user_wins = 0  

```
## Features of the RPS game implementation in the code- 
- When functional play() is called,code will print :
>Game is about to start. Show your choice in front of the camera (Rock, Paper, Scissors)
- After 3 sec wait, frame (224x224) will open to capture the user choice.
- 'Show your choice. Countdown - ' warning message will appear on the frame.
- After 5 sec , code will capture frame and process the frame data and passed it to the trained model (keras_model.h5) to predict the outcome of user choice.
- The max predicted value is then calculated to find the user choice. 
> The max prediction value is 0.9999892711639404

>The predicted list is [[7.55216202e-14 3.22580990e-11 9.99989271e-01 1.07331225e-05]]

> User choice is : Scissors
- If no valid choice is captured, then error message is printed.

> User choice is : Nothing

> There is no choice captured. Please retry.
- Depending on user and computer choices , winner is chosen.

> You lost! Computer's choice is 'Rock'. Your choice is 'Scissors'

> You win! Computer's choice is 'Rock'. Your choice is 'Paper'

>Game draw! Computer's choice is 'Scissors'. Your choice is 'Scissors'

> Computer wins - 2 & User wins - 2
- Game continues till a winner is selected with maximum win of 3.
> Computer wins - 3 & User wins - 2
- After this, user is asked finally if they want to replay the game.
- User is required to enter 'y' to replay the game.
> Do you want to play again (y/n)?n

- If not , then 'Good Bye!' message is printed or else RPS class is called again to replay the game.
> Good Bye!