import random


def generate_problem():
    """Generates a random addition or subtraction problem."""
    num1 = random.randint(0, 100)
    num2 = random.randint(0, 100)
    operation = random.choice(['+', '-'])
    if operation == '-':
        # Ensure the result is not negative
        num1, num2 = max(num1, num2), min(num1, num2)
    return num1, num2, operation


def calculate_answer(num1, num2, operation):
    """Calculates the answer to the problem."""
    return num1 + num2 if operation == '+' else num1 - num2


def play_game():
    print("Welcome to CalculatePK!!!")
    print("Choose an option:")
    print("S. START")
    print("Q. QUIT")

    while True:
        choice = input("Enter your choice (S/Q): ").strip().upper()
        if choice == 'Q':
            print("Goodbye! Thanks for playing!")
            break
        elif choice == 'S':
            print("GAME START!")
            while True:
                num1, num2, operation = generate_problem()
                print(f"Problem: {num1} {operation} {num2}")
                user_input = input("Your answer (or type 'exit' to quit): ").strip()

                if user_input.lower() == 'exit':
                    print("Exiting game...")
                    break

                if user_input.isdigit() or (user_input.startswith('-') and user_input[1:].isdigit()):
                    user_answer = int(user_input)
                    correct_answer = calculate_answer(num1, num2, operation)
                    if user_answer == correct_answer:
                        print("Correct! Well done!")
                    else:
                        print(f"Incorrect. The correct answer is {correct_answer}.")
                else:
                    print("Invalid input. Please enter a valid number.")
        else:
            print("Invalid choice. Please enter 'S' to start or 'Q' to quit.")


play_game()
