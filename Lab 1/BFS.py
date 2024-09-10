import random
from collections import deque

def bfs_number_guessing_game(min_value, max_value):
    # Initialize the number of attempts
    guess_attempts = 0
    
    # Initialize the queue with the full range of possible numbers (min to max)
    range_queue = deque([(min_value, max_value)])  

    # Instruct the user to think of a number within the specified range
    print(f"Think of a number between {min_value} and {max_value}, and I (the AI) will try to guess it.")

    # Continue the guessing process until the queue is empty or the correct number is guessed
    while range_queue:
        # Dequeue the current range to guess from
        current_min, current_max = range_queue.popleft()
        
        # AI makes a guess randomly within the current range
        ai_guess = random.randint(current_min, current_max)
        print(f"My guess is: {ai_guess}")

        # Increment the number of guesses made
        guess_attempts += 1
        
        # Ask the user for feedback on whether the guess was too high, too low, or correct
        player_feedback = input("Enter 'h' if my guess is too high, 'l' if too low, or 'c' if correct: ").lower()

        if player_feedback == 'c':
            # If the AI guessed correctly, print the number of attempts and exit
            print(f"I guessed your number in {guess_attempts} attempts!")
            return
        elif player_feedback == 'h':
            # If the guess was too high, add a new range to the queue (lower the max bound)
            if current_min <= ai_guess - 1:  # Ensure the new range is valid
                range_queue.append((current_min, ai_guess - 1))
        elif player_feedback == 'l':
            # If the guess was too low, add a new range to the queue (raise the min bound)
            if ai_guess + 1 <= current_max:  # Ensure the new range is valid
                range_queue.append((ai_guess + 1, current_max))
    
    # If the queue is empty and no correct guess was made (unlikely scenario), this prints
    print("I couldn't guess the number. Queue is empty.")

# Start the number guessing game with an initial range of 1 to 100
bfs_number_guessing_game(1, 100)
