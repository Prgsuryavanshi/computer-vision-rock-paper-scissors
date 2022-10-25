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
        u_choice = get_user_choice()

        get_winner(c_choice, u_choice)
        replay = input("Do you want to play again (y/n)?")
        if replay == "y":
            play()
        else:
            print("Good Bye!")
        break    

play()