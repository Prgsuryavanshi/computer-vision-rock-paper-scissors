import time
import random
import cv2
from keras.models import load_model
import numpy as np

class rps():
    def __init__(self,computer_choice, user_choice):
        self.computer_choice = computer_choice
        self.user_choice = user_choice
        self.model = load_model('keras_model.h5')
        self.cap = cv2.VideoCapture(0)
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.computer_wins = 0
        self.user_wins = 0
        

    def get_computer_choice(self):
        '''
        This function will randomly pick an option between "Rock", "Paper", and "Scissors" 
        and return the choice as Computer's.
        '''
        return random.choice(self.computer_choice)

    def get_winner(self,computer_choice, user_choice):
        '''
        This function will choose a winner based on the classic rules of Rock-Paper-Scissors.
        This function takes two arguments: computer_choice and user_choice.
        '''
        
        print(f"The max prediction value is {np.amax(user_choice)}")
        predicted_list_value = self.user_choice[np.argmax(user_choice)]
        print(f'The predicted list from trained model is {user_choice}')
        print(f"User choice is : {predicted_list_value}")
        
        if computer_choice == predicted_list_value:
            print(f"Game draw! Computer's choice is '{computer_choice}'. Your choice is '{predicted_list_value}'")
        elif computer_choice == "Rock" and predicted_list_value == "Paper":
            print(f"You win! Computer's choice is '{computer_choice}'. Your choice is '{predicted_list_value}'")
            self.user_wins += 1   
        elif computer_choice == "Paper" and predicted_list_value == "Rock":
            print(f"You lost! Computer's choice is '{computer_choice}'. Your choice is '{predicted_list_value}'")
            self.computer_wins += 1
        elif computer_choice == "Rock" and predicted_list_value == "Scissors":
            print(f"You lost! Computer's choice is '{computer_choice}'. Your choice is '{predicted_list_value}'")
            self.computer_wins += 1  
        elif computer_choice == "Scissors" and predicted_list_value == "Rock":
            print(f"You win! Computer's choice is '{computer_choice}'. Your choice is '{predicted_list_value}'")
            self.user_wins += 1
        elif computer_choice == "Paper" and predicted_list_value == "Scissors":
            print(f"You win! Computer's choice is '{computer_choice}'. Your choice is '{predicted_list_value}'")
            self.user_wins += 1  
        elif computer_choice == "Scissors" and predicted_list_value == "Paper":
            print(f"You lost! Computer's choice is '{computer_choice}'. Your choice is '{predicted_list_value}'")
            self.computer_wins += 1
        else:
            print("There is no choice captured. Please retry.")

    def countdown(self):
        '''
        The function time.time() returns the time as a floating point number expressed in seconds since the epoch.
        The countdown function gives the user 3 sec to show the choice in front of the camera.
        '''
        print("Game is about to start. Show your choice in front of the camera (Rock, Paper, Scissors)")
        wait_time = 3
        start_time = time.time()
        while (time.time() - start_time) < wait_time:
            pass

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
