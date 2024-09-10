import random

# Instruct the player to think of a number between the specified range
print("Think of a number between 1 and 100, and I (the AI) will try to guess it using a Greedy Search Algorithm.")

def greedy_guessing_game(min_range, max_range):
    
    attempts = 0  # To keep track of the number of attempts made by the AI
    
    # Start with an arbitrary initial guess (midpoint of the range)
    guess = (min_range + max_range) // 2
    
    while True:
        # AI makes a guess
        print(f"My guess is: {guess}")
        attempts += 1

        # Get feedback from the user on the AI's guess
        player_response = input("Enter 'h' if my guess is too high, 'l' if too low, or 'c' if correct: ").lower()

        if player_response == 'c':
            # If the guess is correct, print the number of attempts it took
            print(f"I guessed the number in {attempts} attempts!")
            return
        elif player_response == 'h':
            # If the guess is too high, greedily reduce the range by moving to the lower half
            max_range = guess - 1
        elif player_response == 'l':
            # If the guess is too low, greedily increase the range by moving to the upper half
            min_range = guess + 1
        
        # Greedily select the next guess by moving to the midpoint of the updated range
        guess = (min_range + max_range) // 2

# Start the guessing game using the greedy search algorithm
greedy_guessing_game(1, 100)
