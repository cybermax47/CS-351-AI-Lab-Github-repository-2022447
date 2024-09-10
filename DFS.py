import random

# Instruct the player to think of a number between the specified range
print("Think of a number between 1 and 100, and I (the AI) will try to guess it.")

def dfs_number_guessing_game(min_range, max_range, attempt_count=0):
    """
    This function implements the Depth-First Search (DFS) strategy for the number guessing game.
    
    Parameters:
    min_range (int): The lower bound of the current search range.
    max_range (int): The upper bound of the current search range.
    attempt_count (int): The current number of attempts made by the AI.
    """
    
    # AI makes a guess randomly within the provided range
    ai_guess = random.randint(min_range, max_range)
    print(f"My guess is: {ai_guess}")

    # Increment the number of attempts made by the AI
    attempt_count += 1

    # Get feedback from the user on the AI's guess
    player_response = input("Enter 'h' if my guess is too high, 'l' if too low, or 'c' if correct: ").lower()

    if player_response == 'c':
        # If the guess is correct, print the number of attempts it took
        print(f"I guessed the number in {attempt_count} attempts!")
        return
    elif player_response == 'h':
        # If the guess is too high, continue searching the left half of the current range
        dfs_number_guessing_game(min_range, ai_guess - 1, attempt_count)
    elif player_response == 'l':
        # If the guess is too low, continue searching the right half of the current range
        dfs_number_guessing_game(ai_guess + 1, max_range, attempt_count)

# Start the number guessing game with an initial range of 1 to 100
dfs_number_guessing_game(1, 100)
