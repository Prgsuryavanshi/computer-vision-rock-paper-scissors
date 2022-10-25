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
