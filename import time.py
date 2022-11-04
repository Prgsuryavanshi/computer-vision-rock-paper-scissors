import time
import random
import cv2
from keras.models import load_model
import numpy as np

class rps():
    def __init__(self,computer_choice, user_choice)
    self.computer_choice = ["Rock", "Paper", "Scissors"]
    self.user_choice = ["Rock", "Paper", "Scissors", "Nothing"]
model = load_model('keras_model.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)   
    
def get_computer_choice():
    '''
    This function will randomly pick an option between "Rock", "Paper", and "Scissors" 
    and return the choice.
    '''
    game = ["Rock", "Paper", "Scissors"]
    return random.choice(game)

def get_winner(computer_choice, user_choice):
    '''
    This function will choose a winner based on the classic rules of Rock-Paper-Scissors.
    This function takes two arguments: computer_choice and user_choice.
    '''
    prediction_list = ["Rock","Paper","Scissors","Nothing"]
    print(f"The max prediction value is {np.amax(user_choice)}")
    predicted_list_value = prediction_list[np.argmax(user_choice)]
    print(f'The predicted list is {user_choice}')
    print(f"User choice is : {predicted_list_value}")
    
    if computer_choice == predicted_list_value:
        print(f"Game draw! Computer's choice is '{computer_choice}'. Your choice is '{predicted_list_value}'")
    elif computer_choice == "Rock" and predicted_list_value == "Paper":
        print(f"You win! Computer's choice is '{computer_choice}'. Your choice is '{predicted_list_value}'")   
    elif computer_choice == "Paper" and predicted_list_value == "Rock":
        print(f"You lost! Computer's choice is '{computer_choice}'. Your choice is '{predicted_list_value}'")
    elif computer_choice == "Rock" and predicted_list_value == "Scissors":
        print(f"You lost! Computer's choice is '{computer_choice}'. Your choice is '{predicted_list_value}'")   
    elif computer_choice == "Scissors" and predicted_list_value == "Rock":
        print(f"You win! Computer's choice is '{computer_choice}'. Your choice is '{predicted_list_value}'")
    elif computer_choice == "Paper" and predicted_list_value == "Scissors":
        print(f"You win! Computer's choice is '{computer_choice}'. Your choice is '{predicted_list_value}'")   
    elif computer_choice == "Scissors" and predicted_list_value == "Paper":
        print(f"You lost! Computer's choice is '{computer_choice}'. Your choice is '{predicted_list_value}'")
    else:
        print("There is no choice captured. Please retry.")
def get_prediction():
    '''
    This function return the output of the model trained for RPS game.
    '''
    start = time.time()
    while (time.time()-start) < 1: 
        ret, frame = cap.read()
        resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
        data[0] = normalized_image
        prediction = model.predict(data)
        cv2.imshow('frame', frame)
        # Press q to close the window
        print(prediction)
        end = time.time()
        print(end - start)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return prediction


def play():
    '''
    Inside this function, following three functions are called : 
    get_computer_choice, 
    get_user_choice, 
    get_winner
    When this function is called, User play a game of Rock-Paper-Scissors, 
    and it prints whether the computer or User won.
    '''
    while True:
        c_choice = get_computer_choice()
        u_choice = get_prediction()

        get_winner(c_choice, u_choice)
        replay = input("Do you want to play again (y/n)?")
        if replay == "y":
            play()
        else:
            print("Good Bye!")
        break 
    # After the loop release the cap object
    cap.release()   
    # Destroy all the windows
    cv2.destroyAllWindows()   

play()
